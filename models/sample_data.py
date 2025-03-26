import numpy as np
from sklearn.model_selection import train_test_split

def generate_diabetes_data(n_samples=1000):
    """Generate synthetic diabetes dataset"""
    np.random.seed(42)
    
    # Generate features
    glucose = np.random.normal(loc=120, scale=30, size=n_samples)
    blood_pressure = np.random.normal(loc=80, scale=10, size=n_samples)
    bmi = np.random.normal(loc=27, scale=5, size=n_samples)
    age = np.random.normal(loc=50, scale=15, size=n_samples)
    insulin = np.random.normal(loc=120, scale=40, size=n_samples)
    skin_thickness = np.random.normal(loc=25, scale=8, size=n_samples)
    pregnancies = np.random.poisson(lam=2, size=n_samples)
    dpf = np.random.normal(loc=0.5, scale=0.3, size=n_samples)  # Diabetes pedigree function
    
    # Combine features
    X = np.column_stack([
        glucose, blood_pressure, bmi, age,
        insulin, skin_thickness, pregnancies, dpf
    ])
    
    # Generate target (simplified logic for demonstration)
    y = (glucose > 140) & (bmi > 30) | (age > 65) & (blood_pressure > 90)
    y = y.astype(int)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def generate_cancer_data(n_samples=1000):
    """Generate synthetic breast cancer dataset"""
    np.random.seed(42)
    
    # Generate 30 features typical in cancer diagnosis
    features = []
    
    # Cell size and shape features
    mean_radius = np.random.normal(loc=15, scale=3, size=n_samples)
    mean_texture = np.random.normal(loc=20, scale=4, size=n_samples)
    mean_perimeter = np.random.normal(loc=90, scale=15, size=n_samples)
    mean_area = np.random.normal(loc=600, scale=100, size=n_samples)
    
    # Generate additional features
    for i in range(26):  # Additional features to make total of 30
        features.append(np.random.normal(loc=50, scale=10, size=n_samples))
    
    # Combine all features
    X = np.column_stack([mean_radius, mean_texture, mean_perimeter, mean_area] + features)
    
    # Generate target (simplified logic for demonstration)
    y = (mean_radius > 17) & (mean_area > 650) | (mean_texture > 25)
    y = y.astype(int)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def generate_heart_data(n_samples=1000):
    """Generate synthetic heart disease dataset"""
    np.random.seed(42)
    
    # Generate features
    age = np.random.normal(loc=55, scale=10, size=n_samples)
    resting_bp = np.random.normal(loc=130, scale=20, size=n_samples)
    cholesterol = np.random.normal(loc=220, scale=40, size=n_samples)
    max_heart_rate = np.random.normal(loc=150, scale=20, size=n_samples)
    st_depression = np.random.normal(loc=1, scale=0.5, size=n_samples)
    
    # Categorical features
    chest_pain = np.random.randint(0, 4, size=n_samples)  # 4 types
    rest_ecg = np.random.randint(0, 3, size=n_samples)    # 3 types
    angina = np.random.randint(0, 2, size=n_samples)      # Binary
    st_slope = np.random.randint(0, 3, size=n_samples)    # 3 types
    vessels = np.random.randint(0, 4, size=n_samples)     # 0-3 vessels
    thal = np.random.randint(0, 3, size=n_samples)        # 3 types
    
    # Combine features
    X = np.column_stack([
        age, resting_bp, cholesterol, max_heart_rate, st_depression,
        chest_pain, rest_ecg, angina, st_slope, vessels, thal
    ])
    
    # Generate target (simplified logic for demonstration)
    y = ((age > 60) & (resting_bp > 140)) | \
        ((cholesterol > 250) & (max_heart_rate > 170)) | \
        ((st_depression > 1.5) & (angina == 1))
    y = y.astype(int)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(diabetes_model, cancer_model, heart_model):
    """Train all models with sample data"""
    # Train diabetes model
    X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = generate_diabetes_data()
    diabetes_model.train(X_train_diabetes, y_train_diabetes)
    diabetes_accuracy = diabetes_model.evaluate(X_test_diabetes, y_test_diabetes)
    
    # Train cancer model
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = generate_cancer_data()
    cancer_model.train(X_train_cancer, y_train_cancer)
    cancer_accuracy = cancer_model.evaluate(X_test_cancer, y_test_cancer)
    
    # Train heart disease model
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = generate_heart_data()
    heart_model.train(X_train_heart, y_train_heart)
    heart_accuracy = heart_model.evaluate(X_test_heart, y_test_heart)
    
    return {
        'diabetes': {
            'accuracy': diabetes_accuracy,
            'n_samples': len(X_train_diabetes) + len(X_test_diabetes)
        },
        'cancer': {
            'accuracy': cancer_accuracy,
            'n_samples': len(X_train_cancer) + len(X_test_cancer)
        },
        'heart': {
            'accuracy': heart_accuracy,
            'n_samples': len(X_train_heart) + len(X_test_heart)
        }
    } 