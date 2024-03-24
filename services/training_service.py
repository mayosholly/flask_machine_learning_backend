# services/training_service.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data.db_connection import fetch_data_from_mysql, fetch_heart_data_from_mysql

def train_model():
    # Fetch data from MySQL
    data = fetch_data_from_mysql()

    # Assume 'Outcome' is the target column
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Save the trained model
    joblib.dump(model, 'diabeticModel/lrmodel_retrained.joblib')

    return {'message': 'Model retrained successfully', 'accuracy': accuracy}


def train_heart_model():
    # Fetch data from MySQL
    data = fetch_heart_data_from_mysql()

    # Assume 'Outcome' is the target column
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Save the trained model
    joblib.dump(model, 'heartModel/heartModel_retrained.joblib')

    return {'message': 'Model retrained successfully', 'accuracy': accuracy}