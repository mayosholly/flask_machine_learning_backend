# models/prediction_model.py
import joblib

def load_prediction_model(model_path):
    return joblib.load(model_path)


def load_prediction_model_ln(model_path):
    return joblib.load(model_path)


def load_prediction_model_et(model_path):
    return joblib.load(model_path)


def load_prediction_model_rf(model_path):
    return joblib.load(model_path)


def load_prediction_model_dt(model_path):
    return joblib.load(model_path)
