# controllers/prediction_controller.py
from flask import jsonify
from services.heart_prediction_service import extract_features
from models.heart_prediction_model import load_prediction_model
import numpy as np
from services.training_service import train_heart_model


# model = load_prediction_model()

def validate_input(data):
    required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak','slope','ca', 'thal']

    for field in required_fields:
        if field not in data:
            raise ValueError(f'Missing required field: {field}')

        value = data[field]

        # Advanced validation for each field
        if field == 'age':
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'sex':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'cp':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'trestbps':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'chol':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'fbs':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'restecg':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'thalach':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
        elif field == 'exang':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
        elif field == 'oldpeak':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
        elif field == 'slope':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
        elif field == 'ca':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
        elif field == 'thal':
            if not isinstance(value,  (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')
    return True

def predict_heart(data, model_path):
    model = load_prediction_model(model_path)
    # Validate input
    validate_input(data)

    # Extract the features
    features = extract_features(data)

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Prepare the response
    return {'prediction': int(prediction[0])}

def retrain_heart():
    try:
        result = train_heart_model()
        return result

    except Exception as e:
        return jsonify({'error': str(e)}), 500