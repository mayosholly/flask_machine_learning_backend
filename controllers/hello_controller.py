# controllers/hello_controller.py
from flask import jsonify

def hello_world():
    return jsonify({'message': 'Hello, World!'})
