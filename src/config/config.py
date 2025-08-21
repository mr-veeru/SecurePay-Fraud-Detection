"""
SecurePay Fraud Detection - Configuration
----------------------------------------
This module provides central configuration for all components.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Model files
MODEL_FILES = {
    'logistic_regression': os.path.join(MODELS_DIR, "logistic_regression.pkl"),
    'random_forest': os.path.join(MODELS_DIR, "random_forest.pkl"),
    'xgboost': os.path.join(MODELS_DIR, "xgboost.pkl"),
    'scaler': os.path.join(MODELS_DIR, "scaler.pkl"),
    'feature_names': os.path.join(MODELS_DIR, "feature_names.json")
}

# API settings
API_HOST = os.getenv('API_HOST', "0.0.0.0")
API_PORT = int(os.getenv('API_PORT', "5000"))
API_URL = f"http://{API_HOST}:{API_PORT}"
MAX_API_CALLS = int(os.getenv('MAX_API_CALLS', "100"))
API_RATE_LIMIT_WINDOW = int(os.getenv('API_RATE_LIMIT_WINDOW', "60"))

# Feature engineering settings
TARGET_COLUMN = "is_fraud"
BALANCE_METHOD = os.getenv('BALANCE_METHOD', "auto")
FEATURE_SELECTION_METHOD = os.getenv('FEATURE_SELECTION_METHOD', "mutual_info")
NUM_FEATURES_TO_SELECT = int(os.getenv('NUM_FEATURES_TO_SELECT', "15"))

# Model training settings
RANDOM_SEED = int(os.getenv('RANDOM_SEED', "42"))
TEST_SIZE = float(os.getenv('TEST_SIZE', "0.2"))
CV_FOLDS = int(os.getenv('CV_FOLDS', "5"))

# Model hyperparameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': int(os.getenv('RF_N_ESTIMATORS', "100")),
        'max_depth': int(os.getenv('RF_MAX_DEPTH', "20"))
    },
    'xgboost': {
        'learning_rate': float(os.getenv('XGB_LEARNING_RATE', "0.1")),
        'n_estimators': int(os.getenv('XGB_N_ESTIMATORS', "100"))
    },
    'logistic_regression': {
        'C': float(os.getenv('LR_C', "1.0")),
        'max_iter': int(os.getenv('LR_MAX_ITER', "1000"))
    }
}

# Dashboard settings
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', "0.0.0.0")
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', "8501"))
MAX_DEMO_SAMPLES = int(os.getenv('MAX_DEMO_SAMPLES', "1000"))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', "INFO")
LOG_FORMAT = os.getenv('LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create required directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True) 