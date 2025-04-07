"""
SecurePay Fraud Detection - Common Utilities
--------------------------------------------
This module provides common utility functions used across the project.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))
from src.config.config import LOGS_DIR, LOG_FORMAT, LOG_LEVEL

# State and gender mappings for dashboard
US_STATE_MAPPING = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
    'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
    'DC': 'District of Columbia'
}

GENDER_MAPPING = {
    0: 'Male',
    1: 'Female',
    'M': 'Male',
    'F': 'Female'
}

def setup_logging(logger_name, log_file=None):
    """
    Set up logging configuration.
    
    Parameters:
    -----------
    logger_name : str
        Name of the logger
    log_file : str, optional
        Name of the log file. If None, logs will only be sent to console
        
    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL))
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def print_header(title):
    """Print a formatted header for console output."""
    width = 80
    print("\n" + "="*width)
    print(f"{title:^{width}}")
    print("="*width + "\n")

def load_creditcard_data():
    """
    Load the credit card dataset from various possible locations.
    
    Returns:
    --------
    pandas.DataFrame or None
        The loaded data or None if no data is found
    """
    # Define possible file paths
    possible_paths = [
        'data/raw/creditcard.csv',
        'data/creditcard.csv',
        'creditcard.csv',
        'data/raw/creditcard_sample.csv',
        'data/creditcard_sample.csv'
    ]
    
    # Try to load from each path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from {path}")
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"Error loading data from {path}: {e}")
    
    # If no data is found, return None
    print("No credit card data found. Please download the dataset.")
    return None

def apply_mappings(df):
    """Apply standard mappings to make categorical features more readable."""
    # Create a copy of the dataframe to avoid modifying the original
    df_mapped = df.copy()
    
    # Apply gender mapping if the column exists
    if 'gender' in df_mapped.columns:
        gender_map = GENDER_MAPPING
        try:
            df_mapped['gender'] = df_mapped['gender'].map(gender_map).fillna(df_mapped['gender'])
        except:
            # If mapping fails (e.g., if already mapped), keep original values
            pass
    
    # Apply state mapping if the column exists
    if 'state' in df_mapped.columns:
        state_map = US_STATE_MAPPING
        try:
            # Add a display column for states while keeping the original codes
            df_mapped['state_display'] = df_mapped['state'].map(state_map).fillna(df_mapped['state'])
        except:
            # If mapping fails, keep original values
            pass
    
    return df_mapped

def save_model_results(model_name, metrics, feature_importance=None):
    """
    Save model metrics and feature importance to a file.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    metrics : dict
        Dictionary of model metrics
    feature_importance : pandas.DataFrame, optional
        Feature importance data
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save metrics
    metrics_file = f'models/{model_name}_metrics.txt'
    with open(metrics_file, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    # Save feature importance if available
    if feature_importance is not None:
        importance_file = f'models/{model_name}_importance.csv'
        feature_importance.to_csv(importance_file, index=False)

def weighted_average_ensemble(predictions, weights=None):
    """
    Create a weighted average ensemble of predictions.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary of model predictions {model_name: predictions}
    weights : dict, optional
        Dictionary of model weights {model_name: weight}
        
    Returns:
    --------
    numpy.ndarray
        Weighted average predictions
    """
    if weights is None:
        # Equal weights if not specified
        weights = {model: 1/len(predictions) for model in predictions}
    
    # Calculate weighted sum
    weighted_sum = np.zeros_like(list(predictions.values())[0])
    
    for model, preds in predictions.items():
        if model in weights:
            weighted_sum += weights[model] * preds
    
    return weighted_sum 