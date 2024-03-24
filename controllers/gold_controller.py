# controllers/prediction_controller.py
from flask import jsonify
from services.gold_prediction import extract_features, extract_features_dt, extract_features_et,extract_features_ln,extract_features_rt
from models.gold_prediction import load_prediction_model, load_prediction_model_dt,load_prediction_model_et,load_prediction_model_ln,load_prediction_model_rf
import numpy as np
# from services.training_service import train_model



# model = load_prediction_model()

def validate_input(data):
    required_fields = ['Open', 'High', 'Low', 'Volume']

    for field in required_fields:
        if field not in data:
            raise ValueError(f'Missing required field: {field}')

        value = data[field]

        # Advanced validation for each field
        if field == 'High':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Low':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Volume':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')


    return True


def validate_input_3(data):
    required_fields = ['Open', 'High', 'Low']

    for field in required_fields:
        if field not in data:
            raise ValueError(f'Missing required field: {field}')

        value = data[field]

        # Advanced validation for each field
        if field == 'High':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')

        elif field == 'Low':
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f'Invalid value for {field}')


    return True


def predict_gold(data, model_path):
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




def predict_gold_ln(data, model_path):
    # Load the model specified by the model_path
    model = load_prediction_model_ln(model_path)

    # Validate input
    validate_input_3(data)

    # Extract the features
    features = extract_features_ln(data)

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Prepare the response
    return {'prediction': int(prediction[0])}



def predict_gold_rf(data, model_path):
    # Load the model specified by the model_path
    model = load_prediction_model_rf(model_path)

    # Validate input
    validate_input_3(data)

    # Extract the features
    features = extract_features_rt(data)

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Prepare the response
    return {'prediction': int(prediction[0])}




def predict_gold_dt(data, model_path):
    # Load the model specified by the model_path
    model = load_prediction_model_dt(model_path)

    # Validate input
    validate_input_3(data)

    # Extract the features
    features = extract_features_dt(data)

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Prepare the response
    return {'prediction': int(prediction[0])}



def predict_gold_et(data, model_path):
    # Load the model specified by the model_path
    model = load_prediction_model_et(model_path)

    # Validate input
    validate_input_3(data)

    # Extract the features
    features = extract_features_et(data)

    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features_array)

    # Prepare the response
    return {'prediction': int(prediction[0])}




# def retrain():
#     try:
#         result = train_model()
#         return result

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500