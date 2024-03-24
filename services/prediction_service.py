# services/prediction_service.py
import numpy as np

def extract_features(data):
    return [data[field] for field in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
