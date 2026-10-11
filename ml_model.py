import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime

class StudentPerformancePredictor:
    """
    Machine Learning Model for Student Performance Prediction
    Uses Random Forest and Gradient Boosting ensemble methods
    """
    
    def __init__(self, model_path='models/'):
        """Initialize the predictor with model storage path"""
        self.model_path = model_path
        self.performance_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = [
            'attendance',
            'assignment_marks',
            'internal_exam_marks',
            'study_hours_per_day',
            'previous_semester_percentage',
            'final_exam_marks',
            'previous_marks'
        ]
        self.performance_labels = {
            0: 'Weak',
            1: 'Average',
            2: 'Good',
            3: 'Excellent'
        }
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        # Try to load pre-trained models
        self.load_models()
    
    def _compute_performance_label(self, student_data):
        """
        Compute performance label based on multiple metrics
        Returns: Weak, Average, Good, or Excellent
        """
        attendance = float(student_data.get('attendance', 0))
        assignment = float(student_data.get('assignment_marks', 0))
        internal = float(student_data.get('internal_exam_marks', 0))
        study_hours = float(student_data.get('study_hours_per_day', 0))
        prev_perc = float(student_data.get('previous_semester_percentage', 0))
        final = float(student_data.get('final_exam_marks', 0))
        prev_marks = float(student_data.get('previous_marks', 0))
        
        # Weighted composite score
        composite_score = (
            0.25 * final +
            0.20 * attendance +
            0.15 * prev_perc +
            0.15 * assignment +
            0.15 * internal +
            0.10 * (study_hours * 15) +
            0.10 * prev_marks
        )
        
        if composite_score >= 80:
            return 'Excellent'
        elif composite_score >= 65:
            return 'Good'
        elif composite_score >= 50:
            return 'Average'
        else:
            return 'Weak'
    
    def train(self, students_data):
        """
        Train the ML models using student data
        
        Args:
            students_data: List of student dictionaries with performance metrics
        """
        if not students_data or len(students_data) < 5:
            print("Insufficient data for training. Need at least 5 students.")
            return False
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(students_data)
            
            # Verify all required features exist
            if not all(col in df.columns for col in self.feature_columns):
                print(f"Missing required features: {self.feature_columns}")
                return False
            
            # Create target labels
            df['performance_label'] = df.apply(self._compute_performance_label, axis=1)
            
            # Prepare features and target
            X = df[self.feature_columns].values.astype(float)
            y = df['performance_label'].values
            
            # Handle any NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Initialize scaler and encoder
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train ensemble model (Random Forest + Gradient Boosting)
            if len(np.unique(y_encoded)) < 2:
                print("Need at least 2 different performance classes")
                return False
            
            self.performance_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=0
            )
            
            self.performance_model.fit(X_scaled, y_encoded)
            
            # Calculate training metrics
            y_pred = self.performance_model.predict(X_scaled)
            accuracy = accuracy_score(y_encoded, y_pred)
            
            print(f"Model trained successfully. Accuracy: {accuracy:.2%}")
            print(f"Performance distribution: {np.bincount(y_encoded)}")
            
            # Save models
            self.save_models()
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict(self, student_data):
        """
        Predict student performance
        
        Args:
            student_data: Dictionary with student performance metrics
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        if self.performance_model is None:
            # Return rule-based prediction if model not trained
            return self._rule_based_prediction(student_data)
        
        try:
            # Prepare input features
            features = []
            for col in self.feature_columns:
                features.append(float(student_data.get(col, 0)))
            
            features = np.array([features]).astype(float)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction and probabilities
            prediction_encoded = self.performance_model.predict(features_scaled)[0]
            probabilities = self.performance_model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            predicted_performance = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = float(np.max(probabilities)) * 100
            
            # Compute grade and result status
            final_marks = float(student_data.get('final_exam_marks', 0))
            grade = self._compute_grade(final_marks)
            result_status = self._compute_result_status(final_marks, float(student_data.get('attendance', 0)))
            
            # Determine risk level
            if predicted_performance == 'Weak' or float(student_data.get('attendance', 0)) < 75:
                risk_level = 'High'
            elif predicted_performance == 'Average':
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Generate improvement suggestions
            suggestions = self._generate_suggestions(student_data, predicted_performance)
            
            return {
                'predicted_performance': predicted_performance,
                'predicted_grade': grade,
                'predicted_result': result_status,
                'confidence_score': confidence,
                'risk_level': risk_level,
                'improvement_suggestions': suggestions,
                'feature_importance': self._get_feature_importance()
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return self._rule_based_prediction(student_data)
    
    def _rule_based_prediction(self, student_data):
        """
        Fallback rule-based prediction when ML model is not available
        """
        performance = self._compute_performance_label(student_data)
        final_marks = float(student_data.get('final_exam_marks', 0))
        grade = self._compute_grade(final_marks)
        attendance = float(student_data.get('attendance', 0))
        result_status = self._compute_result_status(final_marks, attendance)
        
        if performance == 'Weak' or attendance < 75:
            risk_level = 'High'
        elif performance == 'Average':
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'predicted_performance': performance,
            'predicted_grade': grade,
            'predicted_result': result_status,
            'confidence_score': 0.0,
            'risk_level': risk_level,
            'improvement_suggestions': self._generate_suggestions(student_data, performance),
            'feature_importance': {}
        }
    
    def _compute_grade(self, final_marks):
        """Compute letter grade from final marks"""
        final_marks = float(final_marks)
        if final_marks >= 90:
            return 'A+'
        elif final_marks >= 80:
            return 'A'
        elif final_marks >= 70:
            return 'B'
        elif final_marks >= 60:
            return 'C'
        elif final_marks >= 50:
            return 'D'
        elif final_marks >= 40:
            return 'E'
        else:
            return 'F'
    
    def _compute_result_status(self, final_marks, attendance):
        """Determine Pass/Fail status"""
        if float(final_marks) >= 40 and float(attendance) >= 75:
            return 'Pass'
        return 'Fail'
    
    def _generate_suggestions(self, student_data, performance):
        """
        Generate improvement suggestions based on student data and performance
        """
        suggestions = []
        
        attendance = float(student_data.get('attendance', 0))
        if attendance < 75:
            suggestions.append('Improve attendance. Current attendance is below minimum threshold.')
        
        final_marks = float(student_data.get('final_exam_marks', 0))
        if final_marks < 50:
            suggestions.append('Focus on final exam preparation. Consider tutoring or study groups.')
        
        assignment_marks = float(student_data.get('assignment_marks', 0))
        if assignment_marks < 60:
            suggestions.append('Improve assignment submission quality. Complete assignments on time.')
        
        internal_marks = float(student_data.get('internal_exam_marks', 0))
        if internal_marks < 60:
            suggestions.append('Strengthen conceptual understanding through regular practice.')
        
        study_hours = float(student_data.get('study_hours_per_day', 0))
        if study_hours < 2:
            suggestions.append('Increase daily study hours. Aim for at least 2-3 hours of focused study.')
        
        prev_perc = float(student_data.get('previous_semester_percentage', 0))
        if performance in ['Weak', 'Average'] and prev_perc >= 60:
            suggestions.append('Performance declining from previous semester. Analyze areas of difficulty.')
        
        if not suggestions:
            suggestions.append('Maintain current performance level and continue consistent effort.')
        
        return ' | '.join(suggestions)
    
    def _get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.performance_model is None:
            return {}
        
        try:
            importances = self.performance_model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importances))
            # Sort by importance
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        except:
            return {}
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            joblib.dump(self.performance_model, f"{self.model_path}performance_model_{timestamp}.pkl")
            joblib.dump(self.scaler, f"{self.model_path}scaler_{timestamp}.pkl")
            joblib.dump(self.label_encoder, f"{self.model_path}label_encoder_{timestamp}.pkl")
            print(f"Models saved successfully at {timestamp}")
        except Exception as e:
            print(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load most recent trained models from disk"""
        try:
            # Find most recent model files
            if os.path.exists(self.model_path):
                model_files = [f for f in os.listdir(self.model_path) if f.startswith('performance_model_')]
                if model_files:
                    latest_model = sorted(model_files)[-1]
                    base_name = latest_model.replace('performance_model_', '').replace('.pkl', '')
                    
                    self.performance_model = joblib.load(f"{self.model_path}performance_model_{base_name}.pkl")
                    self.scaler = joblib.load(f"{self.model_path}scaler_{base_name}.pkl")
                    self.label_encoder = joblib.load(f"{self.model_path}label_encoder_{base_name}.pkl")
                    print(f"Models loaded successfully from {base_name}")
                    return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
        
        return False
    
    def get_model_info(self):
        """Get information about trained model"""
        if self.performance_model is None:
            return {'status': 'No model trained', 'model_type': None}
        
        return {
            'status': 'Ready',
            'model_type': type(self.performance_model).__name__,
            'feature_columns': self.feature_columns,
            'performance_labels': self.performance_labels,
            'feature_importance': self._get_feature_importance()
        }
