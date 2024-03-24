from flask import Flask, request
from flask_cors import CORS  # Import the CORS module
from controllers.prediction_controller import predict, retrain
from controllers.heart_prediction_controller import predict_heart, train_heart_model
from controllers.gold_controller import predict_gold, predict_gold_dt, predict_gold_et, predict_gold_rf, predict_gold_ln
from controllers.hello_controller import hello_world
from flask import jsonify
from models.auth_models import db
from controllers.auth_controller import auth_bp,token_blacklist
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_jwt_extended import jwt_required, get_jwt_identity

from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

# import matplotlib.pyplot as plt




app = Flask(__name__)

CORS(app)  # Enable CORS for all routes in the app
# CORS(app, origins=['http://allowed-origin.com', 'https://another-allowed-origin.com'])

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/machine_learning'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'

jwt = JWTManager(app)


db.init_app(app)
migrate = Migrate(app, db)

current_model_path = 'diabeticModel/lrmodel.joblib'
current_heart_model_path = 'heartModel/heartModel.joblib'
current_gold_model_path = 'goldModel/ln_model.joblib'
current_gold_model_ln_path = 'goldModel/ln_model3.joblib'
current_gold_model_dt_path = 'goldModel/dt_model3.joblib'
current_gold_model_et_path = 'goldModel/et_model3.joblib'
current_gold_model_rf_path = 'goldModel/rf_model3.joblib'

with app.app_context():
    db.create_all()

app.register_blueprint(auth_bp, url_prefix='/auth')


img_size_224p = 224  # Update with your target image size
model_path = 'orchid/model_fold_5.h5'  # Update with your model path
class_labels = {0: 'Cattleya', 1: 'Dendrobium', 2: 'Oncidium', 3: 'Phalaenopsis', 4: 'Vanda'}

def load_image(filename):
    img = load_img(filename, target_size=(128, 128))  # Update to match the expected input size
    img = img_to_array(img)
    img = img.reshape(-1, 128, 128, 3)
    img = img.astype('float32') / 255.0
    return img


def predict_label(img):
    model = load_model(model_path)
    probabilities = model.predict(img)
    predicted_class = np.argmax(probabilities, axis=-1)
    return predicted_class[0]

@app.route('/predict_orchid', methods=['POST'])
def predict_orchid():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            file_path = 'temp.jpg'  # Update with your desired location
            file.save(file_path)

            img = load_image(file_path)
            predicted_class = predict_label(img)
            predicted_label = class_labels.get(predicted_class, 'Unknown')

            return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-dibetics', methods=['POST'])
# @jwt_required()
def predict_route():
    try:
        data = request.get_json(force=True)
        # result = predict(data)

 # Use the most recent model for prediction
        result = predict(data, current_model_path)

        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/predict-heart', methods=['POST'])
def predict_heart_route():
    try:
        data = request.get_json(force=True)
        result = predict_heart(data, current_heart_model_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/predict-gold', methods=['POST'])
def predict_gold_route():
    try:
        data = request.get_json(force=True)
        result = predict_gold(data, current_gold_model_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict-gold-ln', methods=['POST'])
def predict_gold_ln_route():
    try:
        data = request.get_json(force=True)
        result = predict_gold_ln(data, current_gold_model_ln_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-gold-rf', methods=['POST'])
def predict_gold_rf_route():
    try:
        data = request.get_json(force=True)
        result = predict_gold_rf(data, current_gold_model_rf_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict-gold-dt', methods=['POST'])
def predict_gold_dt_route():
    try:
        data = request.get_json(force=True)
        result = predict_gold_dt(data, current_gold_model_dt_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict-gold-et', methods=['POST'])
def predict_gold_et_route():
    try:
        data = request.get_json(force=True)
        result = predict_gold_et(data, current_gold_model_et_path)
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/hello', methods=['GET'])
def hello_route():
    return hello_world()

@app.route('/retrain', methods=['POST'])
def retrain_route():
    global current_model_path  # Use the global keyword to modify the variable

    result = retrain()

    # Update the current model path with the most recent one
    current_model_path = result.get('model_path', current_model_path)

    return jsonify(result)


@app.route('/retrain_heart', methods=['POST'])
def retrain_heart_route():
    global current_heart_model_path  # Use the global keyword to modify the variable

    result = train_heart_model()

    # Update the current model path with the most recent one
    current_heart_model_path = result.get('model_path', current_heart_model_path)

    return jsonify(result)



@jwt.token_in_blocklist_loader
def check_if_token_in_blacklist(jwt_header, jwt_payload):
    jti = jwt_payload['jti']
    return jti in token_blacklist

if __name__ == '__main__':
    app.run(port=5000, debug=True)
