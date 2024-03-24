# controllers.py
from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token
from models.auth_models import db, bcrypt, User
from flask_jwt_extended import jwt_required, get_jwt_identity, decode_token

auth_bp = Blueprint('auth', __name__)


def is_valid_username(username):
    # Add your validation rules for the username here
    return username and len(username) >= 3

def is_valid_password(password):
    # Add your validation rules for the password here
    return password and len(password) >= 6

def is_valid_email(email):
    # Add your validation rules for the email here
    return email and '@' in email

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not is_valid_username(username) or not is_valid_password(password) or not is_valid_email(email):
        return jsonify({'message': 'Invalid username, email, or password format'}), 400

    if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
        return jsonify({'message': 'User with username or email already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not (is_valid_email(email) and is_valid_password(password)):
        return jsonify({'message': 'Invalid email or password format'}), 400

    user = User.query.filter_by(email=email).first()

    if not user or not bcrypt.check_password_hash(user.password, password):
        return jsonify({'message': 'Invalid credentials'}), 401

    # Generate JWT token upon successful login
    access_token = create_access_token(identity=user.username)

    # Return user information along with the access token
    user_info = {
        'username': user.username,
        'email': user.email,
    }

    return jsonify({'message': 'Login successful', 'token': access_token, 'user': user_info}), 200



# Simple in-memory blacklist
token_blacklist = set()

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    # Get the identity of the current user
    current_user = get_jwt_identity()

    # Get the raw JWT token
    raw_token = request.headers.get('Authorization').split(' ')[1]
    
    # Decode the raw token to get the JTI (JWT ID)
    jti = decode_token(raw_token)['jti']

    # Add the token to the blacklist
    token_blacklist.add(jti)

    return jsonify({'message': 'Logout successful', 'user': current_user}), 200