#!/usr/bin/env python3
"""
Simple Fraud Detection API Server
"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global variables for models
models = {}
scaler = None
feature_names = None


def load_models():
    """Load models and dependencies."""
    global scaler, feature_names

    # Load scaler
    try:
        scaler = joblib.load('models/scaler.pkl')
        print("Loaded scaler successfully")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return False

    # Load feature names
    try:
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        print(f"Loaded feature names: {feature_names}")
    except Exception as e:
        print(f"Error loading feature names: {e}")
        return False

    # Load models
    for model_name in ['logistic_regression', 'random_forest']:
        try:
            model_path = f'models/{model_name}.pkl'
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    return len(models) > 0


def preprocess_data(data):
    """Preprocess input data for prediction."""
    if not isinstance(data, (dict, list)):
        raise ValueError("Invalid input format")

    # Convert to list if single instance
    instances = [data] if isinstance(data, dict) else data

    # Extract features in correct order
    features = []
    for instance in instances:
        feature_vector = []
        for feature in feature_names:
            value = instance.get(feature, 0)
            feature_vector.append(
                float(value) if isinstance(value, (int, float)) else 0)
        features.append(feature_vector)

    # Scale features
    return scaler.transform(np.array(features))


def make_prediction(data, model_name='random_forest'):
    """Make predictions using specified model."""
    try:
        processed_data = preprocess_data(data)

        if model_name not in models:
            return None, None

        model = models[model_name]
        pred = model.predict(processed_data)
        prob = model.predict_proba(processed_data)[:, 1] if hasattr(
            model, 'predict_proba') else pred

        return pred, prob

    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None


@app.route('/predict', methods=['POST'])
def predict():
    """Make fraud predictions."""
    try:
        data = request.get_json()
    except:
        return jsonify({'error': 'Invalid JSON'}), 400

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    model_name = request.args.get('model', 'random_forest')
    is_batch = isinstance(data, list)

    predictions, probabilities = make_prediction(data, model_name)
    if predictions is None:
        return jsonify({'error': 'Prediction failed'}), 500

    result = {
        'predictions': predictions.tolist() if is_batch else int(predictions[0]),
        'probabilities': probabilities.tolist() if is_batch else float(probabilities[0]),
        'model_used': model_name,
        'timestamp': datetime.now().isoformat()
    }

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with basic info."""
    return jsonify({
        'message': 'Fraud Detection API',
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict'
        },
        'status': 'running'
    })


if __name__ == '__main__':
    if load_models():
        print("Starting API server...")
        app.run(host='localhost', port=5000)
    else:
        print("Failed to load models. Exiting.")
