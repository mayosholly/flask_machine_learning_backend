# models/prediction_model.py
import joblib

def load_prediction_model(model_path):
    return joblib.load(model_path)
