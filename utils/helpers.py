import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def validate_input_features(features, expected_length, model_type):
    """Validate and preprocess input features"""
    if not isinstance(features, list):
        raise ValueError(f"Features must be a list for {model_type} prediction")
    
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features for {model_type} prediction, got {len(features)}")
    
    try:
        features = np.array(features, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f"All features must be numeric for {model_type} prediction")
    
    return features

def load_scaler(model_type):
    """Load feature scaler if exists"""
    scaler_path = os.path.join('models', f'{model_type}_scaler.joblib')
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return StandardScaler()

def save_scaler(scaler, model_type):
    """Save feature scaler"""
    scaler_path = os.path.join('models', f'{model_type}_scaler.joblib')
    joblib.dump(scaler, scaler_path)

def format_prediction_result(probability):
    """Format prediction result"""
    return {
        'prediction': probability >= 0.5,
        'probability': float(probability)
    }

def validate_model_path(model_path):
    """Validate model file path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return True

def create_model_directory():
    """Create directory for saving models if it doesn't exist"""
    os.makedirs('models/saved', exist_ok=True)

def get_feature_importance(model, feature_names):
    """Get feature importance if available"""
    try:
        importance = model.feature_importances_
        return dict(zip(feature_names, importance))
    except:
        return None 