# controllers/prediction_controller.py
from flask import jsonify
from services.prediction_service import extract_features
from models.prediction_model import load_prediction_model
import numpy as np
from services.training_service import train_model



# model = load_prediction_model()

def validate_input(data):
    required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    for field in required_fields:
        if field not in data:
            raise ValueError(f'Missing required field: {field}')

        value = data[field]

        # Advanced validation for each field
        if field == 'Pregnancies':
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Glucose':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'BloodPressure':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'SkinThickness':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Insulin':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'BMI':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'DiabetesPedigreeFunction':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Age':
            if not isinstance(value, int) or value < 0:
                raise ValueError(f'Invalid value for {field}')

    return True


def predict(data, model_path):
    # Load the model specified by the model_path
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



def retrain():
    try:
        result = train_model()
        return result

    except Exception as e:
        return jsonify({'error': str(e)}), 500