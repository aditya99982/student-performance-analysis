from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for faculty authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    department = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    students = db.relationship('Student', backref='faculty', lazy=True, cascade='all, delete-orphan')
    events = db.relationship('Event', backref='user', lazy=True, cascade='all, delete-orphan')
    predictions = db.relationship('Prediction', backref='faculty', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Student(db.Model):
    """Student model for storing student performance data"""
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False, index=True)
    email = db.Column(db.String(120), nullable=True, index=True)
    department = db.Column(db.String(100), nullable=False, index=True)
    semester = db.Column(db.Integer, nullable=False)
    phone = db.Column(db.String(15), nullable=True)
    
    # Academic Performance Metrics
    marks = db.Column(db.Float, nullable=False, default=0)
    previous_marks = db.Column(db.Float, nullable=False, default=0)
    attendance = db.Column(db.Float, nullable=False, default=0)  # Percentage
    assignment_marks = db.Column(db.Float, nullable=False, default=0)
    internal_exam_marks = db.Column(db.Float, nullable=False, default=0)
    study_hours_per_day = db.Column(db.Float, nullable=False, default=0)
    previous_semester_percentage = db.Column(db.Float, nullable=False, default=0)
    final_exam_marks = db.Column(db.Float, nullable=False, default=0)
    
    # Computed Fields
    result_status = db.Column(db.String(20), default='Pending')  # Pass/Fail
    grade = db.Column(db.String(5), default='N/A')  # A+, A, B, C, D, E, F
    performance_prediction = db.Column(db.String(50), default='Average')  # Excellent, Good, Average, Weak
    
    # Metadata
    faculty_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.student_id}: {self.name}>'
    
    def to_dict(self):
        """Convert student to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'email': self.email,
            'department': self.department,
            'semester': self.semester,
            'phone': self.phone,
            'marks': self.marks,
            'previous_marks': self.previous_marks,
            'attendance': self.attendance,
            'assignment_marks': self.assignment_marks,
            'internal_exam_marks': self.internal_exam_marks,
            'study_hours_per_day': self.study_hours_per_day,
            'previous_semester_percentage': self.previous_semester_percentage,
            'final_exam_marks': self.final_exam_marks,
            'result_status': self.result_status,
            'grade': self.grade,
            'performance_prediction': self.performance_prediction,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Prediction(db.Model):
    """Model for storing ML prediction results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Prediction Results
    predicted_performance = db.Column(db.String(50), nullable=False)  # ML model output
    predicted_grade = db.Column(db.String(5), nullable=False)
    predicted_result = db.Column(db.String(20), nullable=False)  # Pass/Fail
    confidence_score = db.Column(db.Float, nullable=True)  # Confidence percentage
    
    # Additional Metrics
    risk_level = db.Column(db.String(20), nullable=True)  # High, Medium, Low
    improvement_suggestions = db.Column(db.Text, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.predicted_performance}>'
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'faculty_id': self.faculty_id,
            'predicted_performance': self.predicted_performance,
            'predicted_grade': self.predicted_grade,
            'predicted_result': self.predicted_result,
            'confidence_score': self.confidence_score,
            'risk_level': self.risk_level,
            'improvement_suggestions': self.improvement_suggestions,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Event(db.Model):
    """Model for tracking application events (incoming/outgoing)"""
    __tablename__ = 'events'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    event_type = db.Column(db.String(50), nullable=False)  # incoming, outgoing, system
    event_category = db.Column(db.String(100), nullable=False)  # login, logout, student_added, prediction, etc.
    message = db.Column(db.Text, nullable=False)
    details = db.Column(db.Text, nullable=True)  # JSON string with additional details
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<Event {self.event_type}: {self.event_category}>'
    
    def to_dict(self):
        """Convert event to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'event_type': self.event_type,
            'event_category': self.event_category,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class Analytics(db.Model):
    """Model for storing analytics snapshots"""
    __tablename__ = 'analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    faculty_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    total_students = db.Column(db.Integer, default=0)
    average_final_marks = db.Column(db.Float, default=0)
    average_attendance = db.Column(db.Float, default=0)
    pass_count = db.Column(db.Integer, default=0)
    fail_count = db.Column(db.Integer, default=0)
    excellent_count = db.Column(db.Integer, default=0)
    good_count = db.Column(db.Integer, default=0)
    average_count = db.Column(db.Integer, default=0)
    weak_count = db.Column(db.Integer, default=0)
    low_attendance_count = db.Column(db.Integer, default=0)
    low_marks_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Analytics {self.faculty_id}>'
    
    def to_dict(self):
        """Convert analytics to dictionary"""
        return {
            'id': self.id,
            'faculty_id': self.faculty_id,
            'total_students': self.total_students,
            'average_final_marks': self.average_final_marks,
            'average_attendance': self.average_attendance,
            'pass_count': self.pass_count,
            'fail_count': self.fail_count,
            'excellent_count': self.excellent_count,
            'good_count': self.good_count,
            'average_count': self.average_count,
            'weak_count': self.weak_count,
            'low_attendance_count': self.low_attendance_count,
            'low_marks_count': self.low_marks_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
