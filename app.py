from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import numpy as np
from models.diabetes_model import DiabetesModel
from models.cancer_model import CancerModel
from models.heart_model import HeartModel
from models.sample_data import train_models
from models.database import db, User, Prediction, ModelMetrics, TrainingHistory, init_db
from utils.helpers import (
    validate_input_features,
    format_prediction_result,
    create_model_directory
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Flask app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///medical.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize database
init_db(app)

# Initialize global variables for models
diabetes_model = None
cancer_model = None
heart_model = None
training_stats = None

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_models():
    """Load or initialize ML models"""
    global diabetes_model, cancer_model, heart_model, training_stats
    
    # Create models directory if it doesn't exist
    create_model_directory()
    
    # Initialize models
    diabetes_model = DiabetesModel()
    cancer_model = CancerModel()
    heart_model = HeartModel()
    
    # Train models with sample data
    training_stats = train_models(diabetes_model, cancer_model, heart_model)
    
    # Update model metrics in database
    model_dict = {
        'diabetes': diabetes_model,
        'cancer': cancer_model,
        'heart': heart_model
    }
    
    with app.app_context():
        for model_type, stats in training_stats.items():
            metric = ModelMetrics.query.filter_by(model_type=model_type).first()
            if not metric:
                metric = ModelMetrics(model_type=model_type)
            
            metric.accuracy = stats['accuracy']
            metric.n_samples = stats['n_samples']
            if hasattr(model_dict[model_type].model, 'feature_importances_'):
                metric.feature_importance = model_dict[model_type].model.feature_importances_.tolist()
            
            db.session.add(metric)
        
        db.session.commit()
    
    print("Models trained successfully!")
    print(f"Diabetes model accuracy: {training_stats['diabetes']['accuracy']:.2f}")
    print(f"Cancer model accuracy: {training_stats['cancer']['accuracy']:.2f}")
    print(f"Heart disease model accuracy: {training_stats['heart']['accuracy']:.2f}")

@app.route('/')
def home():
    return render_template('index.html', training_stats=training_stats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('home'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/predict/<model_type>', methods=['POST'])
@login_required
def predict(model_type):
    try:
        data = request.get_json()
        
        if model_type == 'diabetes':
            features = validate_input_features(data['features'], 8, model_type)
            probability = diabetes_model.predict(features)
            model = diabetes_model
        elif model_type == 'cancer':
            features = validate_input_features(data['features'], 30, model_type)
            probability = cancer_model.predict(features)
            model = cancer_model
        elif model_type == 'heart':
            features = validate_input_features(data['features'], 11, model_type)
            probability = heart_model.predict(features)
            model = heart_model
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        result = format_prediction_result(probability)
        
        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            model_type=model_type,
            features=data['features'],
            prediction=result['prediction'],
            probability=probability
        )
        db.session.add(prediction)
        db.session.commit()
        
        # Add feature importance if available
        if hasattr(model.model, 'feature_importances_'):
            if model_type == 'diabetes':
                feature_names = [
                    'Glucose', 'Blood Pressure', 'BMI', 'Age',
                    'Insulin', 'Skin Thickness', 'Pregnancies', 'Diabetes Pedigree'
                ]
            elif model_type == 'cancer':
                feature_names = ['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area'] + \
                              [f'Feature_{i}' for i in range(26)]
            else:  # heart
                feature_names = [
                    'Age', 'Resting BP', 'Cholesterol', 'Max Heart Rate', 'ST Depression',
                    'Chest Pain', 'Rest ECG', 'Angina', 'ST Slope', 'Vessels', 'Thal'
                ]
            result['feature_importance'] = dict(zip(feature_names, model.model.feature_importances_))
        
        return jsonify({
            'success': True,
            **result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model-info')
def model_info():
    metrics = {
        metric.model_type: {
            'accuracy': metric.accuracy,
            'n_samples': metric.n_samples,
            'last_trained': metric.last_trained.isoformat(),
            'feature_importance': metric.feature_importance
        }
        for metric in ModelMetrics.query.all()
    }
    
    return jsonify({
        'status': 'healthy',
        'models': metrics
    })

@app.route('/retrain', methods=['POST'])
@login_required
def retrain_models():
    if not current_user.is_admin:
        return jsonify({
            'success': False,
            'error': 'Only administrators can retrain models'
        }), 403
    
    try:
        global training_stats
        training_stats = train_models(diabetes_model, cancer_model, heart_model)
        
        # Record training history
        for model_type, stats in training_stats.items():
            history = TrainingHistory(
                model_type=model_type,
                accuracy=stats['accuracy'],
                n_samples=stats['n_samples'],
                trained_by=current_user.id
            )
            db.session.add(history)
            
            # Update model metrics
            metric = ModelMetrics.query.filter_by(model_type=model_type).first()
            if metric:
                metric.accuracy = stats['accuracy']
                metric.n_samples = stats['n_samples']
                metric.last_trained = history.trained_at
                if hasattr(locals()[f"{model_type}_model"].model, 'feature_importances_'):
                    metric.feature_importance = locals()[f"{model_type}_model"].model.feature_importances_.tolist()
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully',
            'stats': training_stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/user/predictions')
@login_required
def user_predictions():
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('predictions.html', predictions=predictions)

if __name__ == '__main__':
    # Create all database tables
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
        
        # Load models within app context
        load_models()
    
    # Run the application
    app.run(debug=True) 