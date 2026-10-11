from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from config import config
from models import db, User, Student, Prediction, Event, Analytics
from ml_model import StudentPerformancePredictor
import json
from datetime import datetime
from functools import wraps
import pandas as pd

def create_app(config_name='development'):
    """Application factory function"""
    app = Flask(__name__, template_folder='templates')
    app.config.from_object(config[config_name])
    
    # Initialize extensions
    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please login to access this page.'
    
    # Initialize ML model
    ml_predictor = StudentPerformancePredictor()
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    @app.before_request
    def before_request():
        """Before request handler"""
        db.session.commit()
    
    @app.teardown_appcontext
    def shutdown_session(exception=None):
        """Teardown session"""
        db.session.remove()
    
    # ========================
    # AUTHENTICATION ROUTES
    # ========================
    
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            data = request.get_json() if request.is_json else request.form
            
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '').strip()
            first_name = data.get('first_name', '').strip()
            last_name = data.get('last_name', '').strip()
            department = data.get('department', '').strip()
            
            # Validation
            if not all([username, email, password, first_name, last_name]):
                return jsonify({'success': False, 'message': 'All fields are required'}), 400
            
            if User.query.filter_by(username=username).first():
                return jsonify({'success': False, 'message': 'Username already exists'}), 400
            
            if User.query.filter_by(email=email).first():
                return jsonify({'success': False, 'message': 'Email already exists'}), 400
            
            # Create new user
            user = User(
                username=username,
                email=email,
                first_name=first_name,
                last_name=last_name,
                department=department
            )
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            # Log event
            event = Event(
                user_id=user.id,
                event_type='incoming',
                event_category='registration',
                message=f'Faculty {username} registered successfully'
            )
            db.session.add(event)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Registration successful. Please login.'})
        
        return render_template('register.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            data = request.get_json() if request.is_json else request.form
            
            username = data.get('username', '').strip()
            password = data.get('password', '').strip()
            
            if not username or not password:
                return jsonify({'success': False, 'message': 'Username and password required'}), 400
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password) and user.is_active:
                login_user(user)
                
                # Log event
                event = Event(
                    user_id=user.id,
                    event_type='incoming',
                    event_category='login',
                    message=f'Faculty {username} logged in'
                )
                db.session.add(event)
                db.session.commit()
                
                return jsonify({'success': True, 'message': 'Login successful', 'redirect': url_for('dashboard')})
            
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
        
        return render_template('login.html')
    
    @app.route('/logout', methods=['POST'])
    @login_required
    def logout():
        username = current_user.username
        user_id = current_user.id
        
        # Log event
        event = Event(
            user_id=user_id,
            event_type='outgoing',
            event_category='logout',
            message=f'Faculty {username} logged out'
        )
        db.session.add(event)
        db.session.commit()
        
        logout_user()
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    
    # ========================
    # DASHBOARD ROUTES
    # ========================
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template('dashboard.html')
    
    @app.route('/api/analytics')
    @login_required
    def get_analytics():
        """Get analytics for current faculty"""
        students = Student.query.filter_by(faculty_id=current_user.id).all()
        
        if not students:
            return jsonify({
                'total_students': 0,
                'average_final_marks': 0,
                'average_attendance': 0,
                'pass_count': 0,
                'fail_count': 0,
                'excellent_count': 0,
                'good_count': 0,
                'average_count': 0,
                'weak_count': 0,
                'low_attendance_students': [],
                'low_marks_students': [],
                'prediction_distribution': {},
                'grade_distribution': {}
            })
        
        # Calculate metrics
        total_students = len(students)
        final_marks_list = [s.final_exam_marks for s in students]
        attendance_list = [s.attendance for s in students]
        
        average_final_marks = sum(final_marks_list) / total_students if total_students > 0 else 0
        average_attendance = sum(attendance_list) / total_students if total_students > 0 else 0
        
        pass_count = sum(1 for s in students if s.result_status == 'Pass')
        fail_count = sum(1 for s in students if s.result_status == 'Fail')
        
        prediction_dist = {}
        for s in students:
            pred = s.performance_prediction
            prediction_dist[pred] = prediction_dist.get(pred, 0) + 1
        
        grade_dist = {}
        for s in students:
            grade = s.grade
            grade_dist[grade] = grade_dist.get(grade, 0) + 1
        
        # Find students with low marks and attendance
        low_marks_students = [{
            'student_id': s.student_id,
            'name': s.name,
            'final_exam_marks': s.final_exam_marks
        } for s in students if s.final_exam_marks < 50]
        
        low_attendance_students = [{
            'student_id': s.student_id,
            'name': s.name,
            'attendance': s.attendance
        } for s in students if s.attendance < 75]
        
        # Count by prediction
        excellent_count = prediction_dist.get('Excellent', 0)
        good_count = prediction_dist.get('Good', 0)
        average_count = prediction_dist.get('Average', 0)
        weak_count = prediction_dist.get('Weak', 0)
        
        return jsonify({
            'total_students': total_students,
            'average_final_marks': round(average_final_marks, 2),
            'average_attendance': round(average_attendance, 2),
            'pass_count': pass_count,
            'fail_count': fail_count,
            'excellent_count': excellent_count,
            'good_count': good_count,
            'average_count': average_count,
            'weak_count': weak_count,
            'low_attendance_students': low_attendance_students,
            'low_marks_students': low_marks_students,
            'prediction_distribution': prediction_dist,
            'grade_distribution': grade_dist
        })
    
    # ========================
    # STUDENT MANAGEMENT ROUTES
    # ========================
    
    @app.route('/api/students', methods=['GET'])
    @login_required
    def get_students():
        """Get all students for current faculty"""
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        students_query = Student.query.filter_by(faculty_id=current_user.id).paginate(
            page=page, per_page=per_page
        )
        
        students = [s.to_dict() for s in students_query.items]
        
        return jsonify({
            'students': students,
            'total': students_query.total,
            'pages': students_query.pages,
            'current_page': page
        })
    
    @app.route('/api/students/<int:student_id>', methods=['GET'])
    @login_required
    def get_student(student_id):
        """Get specific student details"""
        student = Student.query.get_or_404(student_id)
        
        if student.faculty_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        return jsonify(student.to_dict())
    
    @app.route('/api/students', methods=['POST'])
    @login_required
    def add_student():
        """Add new student with performance data"""
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        required_fields = [
            'student_id', 'name', 'department', 'semester',
            'marks', 'previous_marks', 'attendance',
            'assignment_marks', 'internal_exam_marks',
            'study_hours_per_day', 'previous_semester_percentage',
            'final_exam_marks'
        ]
        
        missing_fields = [f for f in required_fields if f not in data or not str(data[f]).strip()]
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {', '.join(missing_fields)}'
            }), 400
        
        # Check if student_id already exists
        if Student.query.filter_by(student_id=data['student_id']).first():
            return jsonify({
                'success': False,
                'message': f'Student ID {data['student_id']} already exists'
            }), 400
        
        try:
            # Create new student
            student = Student(
                student_id=data['student_id'],
                name=data['name'],
                email=data.get('email'),
                department=data['department'],
                semester=int(data['semester']),
                phone=data.get('phone'),
                marks=float(data['marks']),
                previous_marks=float(data['previous_marks']),
                attendance=float(data['attendance']),
                assignment_marks=float(data['assignment_marks']),
                internal_exam_marks=float(data['internal_exam_marks']),
                study_hours_per_day=float(data['study_hours_per_day']),
                previous_semester_percentage=float(data['previous_semester_percentage']),
                final_exam_marks=float(data['final_exam_marks']),
                faculty_id=current_user.id
            )
            
            # Compute grade and result using ML predictor
            prediction_result = ml_predictor.predict(student.to_dict())
            student.grade = prediction_result['predicted_grade']
            student.result_status = prediction_result['predicted_result']
            student.performance_prediction = prediction_result['predicted_performance']
            
            db.session.add(student)
            db.session.commit()
            
            # Create prediction record
            prediction = Prediction(
                student_id=student.id,
                faculty_id=current_user.id,
                predicted_performance=prediction_result['predicted_performance'],
                predicted_grade=prediction_result['predicted_grade'],
                predicted_result=prediction_result['predicted_result'],
                confidence_score=prediction_result['confidence_score'],
                risk_level=prediction_result['risk_level'],
                improvement_suggestions=prediction_result['improvement_suggestions']
            )
            db.session.add(prediction)
            
            # Log event
            event = Event(
                user_id=current_user.id,
                event_type='incoming',
                event_category='student_added',
                message=f'Student {student.name} (ID: {student.student_id}) added to the system',
                details=json.dumps({'student_id': student.id})
            )
            db.session.add(event)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Student added successfully',
                'student': student.to_dict()
            }), 201
            
        except Exception as e:
            db.session.rollback()
            return jsonify({
                'success': False,
                'message': f'Error adding student: {str(e)}'
            }), 500
    
    @app.route('/api/students/<int:student_id>', methods=['PUT'])
    @login_required
    def update_student(student_id):
        """Update student performance data"""
        student = Student.query.get_or_404(student_id)
        
        if student.faculty_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        try:
            # Update fields
            for field in ['attendance', 'assignment_marks', 'internal_exam_marks',
                         'study_hours_per_day', 'final_exam_marks', 'previous_marks']:
                if field in data:
                    setattr(student, field, float(data[field]))
            
            # Recompute predictions
            prediction_result = ml_predictor.predict(student.to_dict())
            student.grade = prediction_result['predicted_grade']
            student.result_status = prediction_result['predicted_result']
            student.performance_prediction = prediction_result['predicted_performance']
            
            db.session.commit()
            
            # Create new prediction record
            prediction = Prediction(
                student_id=student.id,
                faculty_id=current_user.id,
                predicted_performance=prediction_result['predicted_performance'],
                predicted_grade=prediction_result['predicted_grade'],
                predicted_result=prediction_result['predicted_result'],
                confidence_score=prediction_result['confidence_score'],
                risk_level=prediction_result['risk_level'],
                improvement_suggestions=prediction_result['improvement_suggestions']
            )
            db.session.add(prediction)
            
            # Log event
            event = Event(
                user_id=current_user.id,
                event_type='incoming',
                event_category='student_updated',
                message=f'Student {student.name} (ID: {student.student_id}) performance data updated',
                details=json.dumps({'student_id': student.id})
            )
            db.session.add(event)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Student updated successfully',
                'student': student.to_dict()
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({
                'success': False,
                'message': f'Error updating student: {str(e)}'
            }), 500
    
    @app.route('/api/students/<int:student_id>', methods=['DELETE'])
    @login_required
    def delete_student(student_id):
        """Delete student record"""
        student = Student.query.get_or_404(student_id)
        
        if student.faculty_id != current_user.id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        try:
            # Log event before deletion
            event = Event(
                user_id=current_user.id,
                event_type='outgoing',
                event_category='student_deleted',
                message=f'Student {student.name} (ID: {student.student_id}) record deleted',
                details=json.dumps({'student_id': student.id})
            )
            db.session.add(event)
            
            db.session.delete(student)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Student deleted successfully'
            })
            
        except Exception as e:
            db.session.rollback()
            return jsonify({
                'success': False,
                'message': f'Error deleting student: {str(e)}'
            }), 500
    
    # ========================
    # PREDICTION AND ANALYSIS ROUTES
    # ========================
    
    @app.route('/api/predict', methods=['POST'])
    @login_required
    def predict_performance():
        """Get ML prediction for a student or new data"""
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        required_fields = [
            'attendance', 'assignment_marks', 'internal_exam_marks',
            'study_hours_per_day', 'previous_semester_percentage',
            'final_exam_marks', 'previous_marks'
        ]
        
        # Prepare student data with defaults
        student_data = {}
        for field in required_fields:
            if field in data:
                student_data[field] = float(data[field])
            else:
                student_data[field] = 0.0
        
        try:
            # Get prediction from ML model
            prediction_result = ml_predictor.predict(student_data)
            
            return jsonify({
                'success': True,
                'prediction': prediction_result
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error making prediction: {str(e)}'
            }), 500
    
    @app.route('/api/train-model', methods=['POST'])
    @login_required
    def train_model():
        """Train ML model with current faculty's student data"""
        students = Student.query.filter_by(faculty_id=current_user.id).all()
        
        if len(students) < 5:
            return jsonify({
                'success': False,
                'message': f'Need at least 5 students to train model. Current: {len(students)}'
            }), 400
        
        try:
            # Convert students to dictionaries
            students_data = [s.to_dict() for s in students]
            
            # Train model
            success = ml_predictor.train(students_data)
            
            if success:
                # Log event
                event = Event(
                    user_id=current_user.id,
                    event_type='incoming',
                    event_category='model_trained',
                    message=f'ML model trained with {len(students)} students',
                    details=json.dumps({'student_count': len(students)})
                )
                db.session.add(event)
                db.session.commit()
                
                return jsonify({
                    'success': True,
                    'message': f'Model trained successfully with {len(students)} students',
                    'model_info': ml_predictor.get_model_info()
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Failed to train model'
                }), 500
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error training model: {str(e)}'
            }), 500
    
    @app.route('/api/model-info', methods=['GET'])
    @login_required
    def get_model_info():
        """Get information about trained ML model"""
        return jsonify(ml_predictor.get_model_info())
    
    # ========================
    # EVENT AND HISTORY ROUTES
    # ========================
    
    @app.route('/api/events', methods=['GET'])
    @login_required
    def get_events():
        """Get events for current user"""
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        event_type = request.args.get('type')  # Optional filter
        
        query = Event.query.filter_by(user_id=current_user.id)
        
        if event_type:
            query = query.filter_by(event_type=event_type)
        
        events_query = query.order_by(Event.timestamp.desc()).paginate(
            page=page, per_page=per_page
        )
        
        events = [e.to_dict() for e in events_query.items]
        
        return jsonify({
            'events': events,
            'total': events_query.total,
            'pages': events_query.pages,
            'current_page': page
        })
    
    # ========================
    # DATA EXPORT ROUTES
    # ========================
    
    @app.route('/api/export/students', methods=['GET'])
    @login_required
    def export_students():
        """Export student data as JSON"""
        students = Student.query.filter_by(faculty_id=current_user.id).all()
        
        if not students:
            return jsonify({'error': 'No students found'}), 404
        
        data = [s.to_dict() for s in students]
        
        return jsonify({
            'success': True,
            'data': data,
            'export_timestamp': datetime.utcnow().isoformat(),
            'total_records': len(data)
        })
    
    # ========================
    # ERROR HANDLERS
    # ========================
    
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({'error': 'Unauthorized'}), 401
    
    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({'error': 'Forbidden'}), 403
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500
    
    return app, db, ml_predictor

# Create app instance
if __name__ == '__main__':
    app, db, ml_predictor = create_app('development')
    
    # Create database tables
    with app.app_context():
        db.create_all()
        print("Database tables created successfully!")
    
    # Run application
    app.run(debug=True, host='0.0.0.0', port=5000)
