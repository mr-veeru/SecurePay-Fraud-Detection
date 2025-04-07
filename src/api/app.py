"""
SecurePay Fraud Detection - API Service
---------------------------------------
This module provides a REST API for fraud detection predictions.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
import time
from functools import wraps
from datetime import datetime
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import setup_logging
from config.config import MAX_API_CALLS, API_RATE_LIMIT_WINDOW, API_HOST, API_PORT, LOG_LEVEL, LOG_FORMAT

# Configure logging
logger = setup_logging("api_service", "api_service.log")

# Initialize Flask app
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)  # Fix for running behind proxy
CORS(app)  # Enable Cross-Origin Resource Sharing

# Add security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Rate limiting
class RateLimiter:
    def __init__(self, max_calls=MAX_API_CALLS, time_frame=API_RATE_LIMIT_WINDOW):
        self.max_calls = max_calls  # Max API calls per time_frame
        self.time_frame = time_frame  # Time frame in seconds
        self.calls = {}  # ip -> list of timestamps
    
    def is_allowed(self, ip):
        current_time = time.time()
        if ip not in self.calls:
            self.calls[ip] = []
        
        # Remove timestamps outside time_frame
        self.calls[ip] = [t for t in self.calls[ip] if current_time - t < self.time_frame]
        
        # Check if under limit
        if len(self.calls[ip]) < self.max_calls:
            self.calls[ip].append(current_time)
            return True
        return False

rate_limiter = RateLimiter()

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if not rate_limiter.is_allowed(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        return f(*args, **kwargs)
    return decorated_function

# Request logging middleware
@app.before_request
def log_request():
    # Skip logging for health check endpoints
    if request.path == '/health':
        return
    
    request.start_time = time.time()
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def log_response(response):
    # Skip logging for health check endpoints
    if request.path == '/health':
        return response
    
    duration = time.time() - getattr(request, 'start_time', time.time())
    logger.info(f"Response: {response.status_code} in {duration:.2f}s")
    return response

# Global variables to store loaded models
models = {}
scaler = None
feature_names = None
loaded_models = set()  # Keep track of which models have been loaded

def load_model(model_name):
    """Load a specific model from disk."""
    try:
        logger.info(f"Loading model: {model_name}")
        
        # Check if the model is already loaded
        if model_name in models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        # Map of model names to file paths
        model_paths = {
            'logistic_regression': 'models/logistic_regression.pkl',
            'random_forest': 'models/random_forest.pkl',
            'xgboost': 'models/xgboost.pkl',
        }
        
        # Check if model is supported
        if model_name not in model_paths:
            logger.warning(f"Unsupported model: {model_name}")
            return False
            
        # Check if model file exists
        model_path = model_paths.get(model_name)
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return False
            
        # Load the model
        models[model_name] = joblib.load(model_path)
        loaded_models.add(model_name)
        logger.info(f"Loaded {model_name} model successfully")
        return True
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        return False

def load_models():
    """Load all trained models from disk."""
    global models, scaler, feature_names
    
    logger.info("Loading models...")
    
    success = True
    
    # Try loading core components (scaler and feature names)
    try:
        # Load the scaler
        scaler_path = 'models/scaler.pkl'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler successfully")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")
        
        # Load feature names (assuming we saved them during training)
        feature_names_path = 'models/feature_names.json'
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Loaded feature names: {feature_names}")
        else:
            # Fallback to default feature names from the training data
            feature_names = ['gender', 'age', 'debt', 'priorDefault', 'employed', 
                            'creditScore', 'ZipCode', 'income']
            logger.warning(f"Feature names file not found, using defaults: {feature_names}")
    except Exception as e:
        logger.error(f"Error loading core components: {str(e)}")
        success = False
    
    # Load priority models (don't load all models initially)
    priority_models = ['logistic_regression', 'random_forest']
    
    for model_name in priority_models:
        if not load_model(model_name):
            success = False
    
    # Add model names to loaded_models even if not loaded
    # This helps with availability reporting
    all_models = ['logistic_regression', 'random_forest', 'xgboost']
    
    logger.info(f"Models loaded: {len(loaded_models)} of {len(all_models)}")
    return success and len(loaded_models) > 0

def preprocess_data(data):
    """Preprocess the input data for prediction."""
    try:
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Handle common naming variations in the input
        # Map transaction amount field if needed
        if 'amount' in df.columns and 'amt' not in df.columns and 'amt' in feature_names:
            df['amt'] = df['amount']
        elif 'amt' in df.columns and 'amount' not in df.columns and 'amount' in feature_names:
            df['amount'] = df['amt']
            
        # Check if we have all required features
        missing_features = [feat for feat in feature_names if feat not in df.columns]
        if missing_features:
            logger.warning(f"Missing features in input data: {missing_features}")
            for feat in missing_features:
                df[feat] = 0  # Default value for missing features
        
        # Select only the features used in training
        df = df[feature_names]
        
        # Scale the features
        if scaler is not None:
            df_scaled = scaler.transform(df)
        else:
            df_scaled = df.values  # Fallback if scaler is not available
        
        return df_scaled
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def make_prediction(data, model_name='ensemble'):
    """Make a fraud prediction using the specified model."""
    try:
        # Preprocess the data
        processed_data = preprocess_data(data)
            
        # Load the model if needed and not already loaded
        if model_name != 'ensemble' and model_name not in models:
            if not load_model(model_name):
                logger.error(f"Failed to load model: {model_name}")
                return None, None
            
        # Check if requested model exists now
        if model_name != 'ensemble' and model_name not in models:
            logger.error(f"Unknown model: {model_name}")
            return None, None
            
        # Check if any models are available
        if not models:
            logger.error("No models available for prediction")
            return None, None
        
        if model_name == 'ensemble':
            # Use available models and average predictions
            predictions = []
            probabilities = []
            
            # Load other models if not already loaded
            ensemble_models = ['logistic_regression', 'random_forest', 'xgboost']
                
            # Try to load any missing models
            for name in ensemble_models:
                if name not in models:
                    load_model(name)
            
            for name, model in models.items():
                try:
                    pred = model.predict(processed_data)
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        pred_prob = model.predict_proba(processed_data)[:, 1]
                        probabilities.append(pred_prob)
                    else:
                        # Use predictions as probabilities if actual probabilities not available
                        probabilities.append(pred.astype(float))
                    
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Error making prediction with {name} model: {e}")
                    # Continue with other models
            
            # Check if we have any valid predictions
            if not predictions:
                logger.error("No valid predictions from any model")
                return None, None
                
            # Average the predictions
            avg_pred = np.mean(predictions, axis=0).round().astype(int)
            avg_prob = np.mean(probabilities, axis=0)
            
            return avg_pred[0], avg_prob[0]
            
        else:
            # Use specific model
            model = models[model_name]
            
            # Get the prediction
            pred = model.predict(processed_data)[0]
            
            # Get probability
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(processed_data)[0, 1]
            else:
                prob = float(pred)
                
            return int(pred), prob
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return None, None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Simplified health check with clear status codes
    status = 'ok' if len(models) > 0 else 'degraded'
    return jsonify({
        'status': status,
        'models_loaded': len(models),
        'available_models': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    }), 200 if status == 'ok' else 207  # 207 Multi-Status for degraded service

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
    """Endpoint for making predictions."""
    # Get request data
    data = request.json
    
    if not data:
        return jsonify({
            'error': 'No data provided',
            'timestamp': datetime.now().isoformat()
        }), 400
    
    # Get model name from query parameters (default to ensemble)
    model_name = request.args.get('model', 'ensemble')
    
    # Check if a single instance or multiple instances
    is_single = isinstance(data, dict)
    
    try:
        # Make prediction
        if is_single:
            prediction, probability = make_prediction(data, model_name)
            if prediction is None:
                return jsonify({
                    'error': 'Failed to make prediction',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
            result = {
                'prediction': int(prediction[0]) if prediction else None,
                'fraud_probability': float(probability[0]) if probability else None,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            }
        else:
            predictions, probabilities = make_prediction(data, model_name)
            if predictions is None:
                return jsonify({
                    'error': 'Failed to make prediction',
                    'timestamp': datetime.now().isoformat()
                }), 500
                
            result = {
                'predictions': predictions,
                'fraud_probabilities': probabilities,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/feedback', methods=['POST'])
@rate_limit
def feedback():
    """Endpoint for receiving feedback on predictions."""
    data = request.json
    
    if not data or 'transaction_id' not in data or 'actual_label' not in data:
        return jsonify({
            'error': 'Invalid feedback data'
        }), 400
    
    # In a real system, this would store the feedback for model retraining
    logger.info(f"Received feedback: {data}")
    
    # Create feedback directory if it doesn't exist
    os.makedirs('data/feedback', exist_ok=True)
    
    # Save feedback to a file
    try:
        feedback_file = 'data/feedback/feedback.json'
        
        # Load existing feedback if available
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        
        # Add new feedback
        feedback_data.append({
            'transaction_id': data['transaction_id'],
            'predicted_label': data.get('predicted_label'),
            'actual_label': data['actual_label'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Save updated feedback
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received'
        })
    
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
@rate_limit
def batch_predict():
    """Endpoint for batch predictions."""
    data = request.json
    
    if not data or not isinstance(data, list):
        return jsonify({
            'error': 'Invalid data format. Expected a list of transactions.'
        }), 400
    
    # Get model name from query parameters (default to ensemble)
    model_name = request.args.get('model', 'ensemble')
    
    try:
        # Make predictions
        predictions, probabilities = make_prediction(data, model_name)
        
        if predictions is None:
            return jsonify({
                'error': 'Failed to make prediction'
            }), 500
        
        # Process results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'transaction_id': data[i].get('transaction_id', f'tx_{i}'),
                'prediction': int(pred),
                'fraud_probability': float(prob)
            })
        
        return jsonify({
            'results': results,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("Starting API server...")
        # Use Waitress or Gunicorn in production
        try:
            from waitress import serve
            logger.info("Using Waitress production server")
            serve(app, host=API_HOST, port=API_PORT, threads=4)
        except ImportError:
            try:
                import gunicorn
                logger.info("Using Gunicorn available (use by running 'gunicorn -w 4 -b 0.0.0.0:5000 app:app')")
                # Fall back to Flask dev server for now
                app.run(debug=False, host=API_HOST, port=API_PORT)
            except ImportError:
                logger.warning("Production server not available. Using Flask development server (NOT FOR PRODUCTION)")
                app.run(debug=False, host=API_HOST, port=API_PORT)
    else:
        logger.error("Failed to load models. API server not started.")