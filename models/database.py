from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'diabetes', 'cancer', 'heart'
    features = db.Column(db.JSON, nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ModelMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    n_samples = db.Column(db.Integer, nullable=False)
    last_trained = db.Column(db.DateTime, default=datetime.utcnow)
    feature_importance = db.Column(db.JSON)

class TrainingHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    n_samples = db.Column(db.Integer, nullable=False)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    trained_by = db.Column(db.Integer, db.ForeignKey('user.id'))

def init_db(app):
    """Initialize the database and create tables"""
    db.init_app(app)
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
            admin.set_password('admin123')  # Change this in production
            db.session.add(admin)
            db.session.commit() 