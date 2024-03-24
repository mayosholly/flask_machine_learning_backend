# data/db_connection.py
import mysql.connector
import pandas as pd

def fetch_data_from_mysql():
    # Adjust these connection details
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'machine_learning'
    }

    connection = mysql.connector.connect(**db_config)
    query = "SELECT * FROM diabetics"
    data = pd.read_sql(query, connection)
    connection.close()

    return data


def fetch_heart_data_from_mysql():
    # Adjust these connection details
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'machine_learning'
    }

    connection = mysql.connector.connect(**db_config)
    query = "SELECT * FROM hearts"
    data = pd.read_sql(query, connection)
    connection.close()

    return data
