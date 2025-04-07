"""
SecurePay Fraud Detection - Configuration
----------------------------------------
This module provides central configuration for all components.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    print("python-dotenv not installed. Environment variables from .env will not be loaded.")
    DOTENV_AVAILABLE = False

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = os.getenv('DATA_DIR', os.path.join(PROJECT_ROOT, "data"))
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(PROJECT_ROOT, "models"))
LOGS_DIR = os.getenv('LOGS_DIR', os.path.join(PROJECT_ROOT, "logs"))

# Data directories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback")

# Model files
LOGISTIC_REGRESSION_MODEL = os.path.join(MODELS_DIR, "logistic_regression.pkl")
RANDOM_FOREST_MODEL = os.path.join(MODELS_DIR, "random_forest.pkl")
XGBOOST_MODEL = os.path.join(MODELS_DIR, "xgboost.pkl")
NEURAL_NETWORK_MODEL = os.path.join(MODELS_DIR, "neural_network.h5")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_FILE = os.path.join(MODELS_DIR, "feature_names.json")

# API settings
API_HOST = os.getenv('API_HOST', "0.0.0.0")
API_PORT = int(os.getenv('API_PORT', "5000"))
API_URL = f"http://{API_HOST}:{API_PORT}"
MAX_API_CALLS = int(os.getenv('MAX_API_CALLS', "100"))  # Max API calls per window
API_RATE_LIMIT_WINDOW = int(os.getenv('API_RATE_LIMIT_WINDOW', "60"))  # 60 seconds
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', "ensemble")

# Security settings
JWT_SECRET = os.getenv('JWT_SECRET', "your-secret-key-should-be-long-and-secure")

# Database settings
USE_DATABASE = os.getenv('USE_DATABASE', 'False').lower() == 'true'
DATABASE_URL = os.getenv('DATABASE_URL', '')

# Feature flags
ENABLE_MODEL_RETRAINING = os.getenv('ENABLE_MODEL_RETRAINING', 'False').lower() == 'true'
COLLECT_FEEDBACK = os.getenv('COLLECT_FEEDBACK', 'True').lower() == 'true'

# Feature engineering settings
TARGET_COLUMN = "is_fraud"
BALANCE_METHOD = os.getenv('BALANCE_METHOD', "auto")  # "smote", "adasyn", "random", or "auto"
FEATURE_SELECTION_METHOD = os.getenv('FEATURE_SELECTION_METHOD', "mutual_info")  # "mutual_info" or "f_classif"
NUM_FEATURES_TO_SELECT = int(os.getenv('NUM_FEATURES_TO_SELECT', "15"))
LEAKAGE_CORRELATION_THRESHOLD = float(os.getenv('LEAKAGE_CORRELATION_THRESHOLD', "0.7"))
COLLINEARITY_THRESHOLD = float(os.getenv('COLLINEARITY_THRESHOLD', "0.95"))

# Model training settings
RANDOM_SEED = int(os.getenv('RANDOM_SEED', "42"))
TEST_SIZE = float(os.getenv('TEST_SIZE', "0.2"))
VALIDATION_SIZE = float(os.getenv('VALIDATION_SIZE', "0.25"))  # Percentage of training data to use for validation
CV_FOLDS = int(os.getenv('CV_FOLDS', "5"))

# Model specific parameters
RF_N_ESTIMATORS = int(os.getenv('RF_N_ESTIMATORS', "100"))
RF_MAX_DEPTH = int(os.getenv('RF_MAX_DEPTH', "20"))
XGB_LEARNING_RATE = float(os.getenv('XGB_LEARNING_RATE', "0.1"))
XGB_N_ESTIMATORS = int(os.getenv('XGB_N_ESTIMATORS', "100"))
LR_C = float(os.getenv('LR_C', "1.0"))
LR_MAX_ITER = int(os.getenv('LR_MAX_ITER', "1000"))

ENSEMBLE_WEIGHTS = {
    "logistic_regression": float(os.getenv('ENSEMBLE_WEIGHT_LR', "0.3")),
    "random_forest": float(os.getenv('ENSEMBLE_WEIGHT_RF', "0.4")),
    "xgboost": float(os.getenv('ENSEMBLE_WEIGHT_XGB', "0.3")),
}

# Dashboard settings
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', "8501"))
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', "0.0.0.0")
MAX_DEMO_SAMPLES = int(os.getenv('MAX_DEMO_SAMPLES', "1000"))

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', "INFO")
LOG_FORMAT = os.getenv('LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEEDBACK_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_user_config():
    """Load user configuration from config.json if available."""
    config_file = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                
            # Update globals with user config
            globals().update(user_config)
            print(f"Loaded user configuration from {config_file}")
        except Exception as e:
            print(f"Error loading user configuration: {e}")

# Load user configuration if available
load_user_config()

def get_config_dict() -> Dict[str, Any]:
    """Return the configuration as a dictionary."""
    config_dict = {k: v for k, v in globals().items() 
                  if k.isupper() and not k.startswith('_')}
    return config_dict

def save_config():
    """Save the current configuration to config.json."""
    config_file = os.path.join(PROJECT_ROOT, "config.json")
    try:
        with open(config_file, 'w') as f:
            json.dump(get_config_dict(), f, indent=2, default=str)
        print(f"Saved configuration to {config_file}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False

def generate_env_file():
    """Generate a .env file from the current configuration."""
    env_file = os.path.join(PROJECT_ROOT, ".env")
    try:
        with open(env_file, 'w') as f:
            f.write("# SecurePay Fraud Detection - Environment Configuration\n\n")
            
            # Group settings by category
            categories = {
                "API Configuration": ["API_HOST", "API_PORT", "API_URL"],
                "Security Settings": ["JWT_SECRET", "MAX_API_CALLS", "API_RATE_LIMIT_WINDOW"],
                "Feature Flags": ["ENABLE_MODEL_RETRAINING", "COLLECT_FEEDBACK"],
                "Path Configuration": ["DATA_DIR", "MODELS_DIR", "LOGS_DIR"],
                "Model Parameters": ["DEFAULT_MODEL", "RF_N_ESTIMATORS", "RF_MAX_DEPTH", 
                                      "XGB_LEARNING_RATE", "XGB_N_ESTIMATORS", "LR_C", "LR_MAX_ITER"]
            }
            
            # Write settings by category
            for category, keys in categories.items():
                f.write(f"# {category}\n")
                for key in keys:
                    if key in globals():
                        value = globals()[key]
                        f.write(f"{key}={value}\n")
                f.write("\n")
                
        print(f"Generated .env file at {env_file}")
        return True
    except Exception as e:
        print(f"Error generating .env file: {e}")
        return False 