# services/prediction_service.py
import numpy as np

def extract_features(data):
    return [data[field] for field in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca','thal']]
