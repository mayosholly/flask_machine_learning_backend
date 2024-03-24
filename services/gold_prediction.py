# services/prediction_service.py
import numpy as np

def extract_features(data):
    return [data[field] for field in ['Open', 'High', 'Low', 'Volume']]


def extract_features_ln(data):
    return [data[field] for field in ['Open', 'High', 'Low']]


def extract_features_rt(data):
    return [data[field] for field in ['Open', 'High', 'Low']]



def extract_features_dt(data):
    return [data[field] for field in ['Open', 'High', 'Low']]


def extract_features_et(data):
    return [data[field] for field in ['Open', 'High', 'Low']]
