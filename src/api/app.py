"""
SecurePay Fraud Detection - API Server
------------------------------------
This module provides the REST API for fraud detection.
"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import API_HOST, API_PORT, MAX_API_CALLS, API_RATE_LIMIT_WINDOW

# Configure logging
logging.basicConfig(
    filename='logs/api_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_service')

app = Flask(__name__)

# Global variables
models = {}
scaler = None
feature_names = None

def rate_limit(f):
    """Rate limiting decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Simple rate limiting based on time window
        now = datetime.now()
        window_start = now.replace(second=0, microsecond=0)
        
        # Get client IP
        client_ip = request.remote_addr
        
        # Check rate limit (implement proper storage in production)
        if getattr(app, 'rate_limit_count', {}).get(client_ip, 0) >= MAX_API_CALLS:
            return jsonify({'error': 'Rate limit exceeded'}), 429
            
        # Update count
        app.rate_limit_count = getattr(app, 'rate_limit_count', {})
        app.rate_limit_count[client_ip] = app.rate_limit_count.get(client_ip, 0) + 1
        
        return f(*args, **kwargs)
    return decorated_function

def load_model(model_name):
    """Load a specific model."""
    if model_name in models:
        return True
        
    model_path = f'models/{model_name}.pkl'
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return False
        
    try:
        models[model_name] = joblib.load(model_path)
        logger.info(f"Loaded {model_name} model successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return False

def load_models():
    """Load models and dependencies."""
    global scaler, feature_names
    
    # Load scaler
    try:
        scaler = joblib.load('models/scaler.pkl')
        logger.info("Loaded scaler successfully")
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        return False
    
    # Load feature names
    try:
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        logger.info(f"Loaded feature names: {feature_names}")
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        return False
    
    # Load priority models
    for model_name in ['logistic_regression', 'random_forest']:
        if not load_model(model_name):
            continue
    
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
            feature_vector.append(float(value) if isinstance(value, (int, float)) else 0)
        features.append(feature_vector)
    
    # Scale features
    return scaler.transform(np.array(features))

def make_prediction(data, model_name='ensemble'):
    """Make predictions using specified model."""
    try:
        processed_data = preprocess_data(data)
        
        if model_name == 'ensemble':
            predictions = []
            probabilities = []
            
            for name, model in models.items():
                if not model:
                    continue
                    
                pred = model.predict(processed_data)
                prob = model.predict_proba(processed_data)[:, 1] if hasattr(model, 'predict_proba') else pred
                
                predictions.append(pred)
                probabilities.append(prob)
            
            if not predictions:
                return None, None
                
            final_pred = np.mean(predictions, axis=0).round().astype(int)
            final_prob = np.mean(probabilities, axis=0)
            
            return final_pred, final_prob
            
        else:
            if model_name not in models and not load_model(model_name):
                return None, None
                
            model = models[model_name]
            pred = model.predict(processed_data)
            prob = model.predict_proba(processed_data)[:, 1] if hasattr(model, 'predict_proba') else pred
            
            return pred, prob
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, None

@app.route('/models', methods=['GET'])
@rate_limit
def list_models():
    """List available models."""
    return jsonify({
        'models': list(models.keys()) + ['ensemble'],
        'default': 'ensemble'
    })

@app.route('/predict', methods=['POST'])
@rate_limit
def predict():
    """Make fraud predictions."""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    model_name = request.args.get('model', 'ensemble')
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

if __name__ == '__main__':
    if load_models():
        logger.info("Starting API server...")
        try:
            from waitress import serve
            serve(app, host=API_HOST, port=API_PORT)
        except ImportError:
            app.run(host=API_HOST, port=API_PORT)