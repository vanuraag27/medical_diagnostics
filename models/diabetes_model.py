import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class DiabetesModel:
    def __init__(self):
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        """Build the diabetes prediction model"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def preprocess_features(self, features):
        """Preprocess input features"""
        # Convert to numpy array
        features = np.array(features).reshape(1, -1)
        
        # Scale features if scaler is fitted
        if hasattr(self.scaler, 'mean_'):
            features = self.scaler.transform(features)
        
        return features
    
    def predict(self, features):
        """Make prediction for given features"""
        processed_features = self.preprocess_features(features)
        prediction = self.model.predict_proba(processed_features)
        return float(prediction[0][1])  # Return probability of positive class
    
    def train(self, X_train, y_train):
        """Train the model with given data"""
        # Fit scaler
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.score(X_test_scaled, y_test)
    
    def save(self, filepath):
        """Save model"""
        import joblib
        joblib.dump((self.model, self.scaler), filepath)
    
    def load(self, filepath):
        """Load model"""
        import joblib
        self.model, self.scaler = joblib.load(filepath) 