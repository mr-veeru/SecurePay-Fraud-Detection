"""
SecurePay Fraud Detection - Model Training
-----------------------------------------
This module handles the training of fraud detection models.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import print_header, load_creditcard_data, apply_mappings
from config.config import RANDOM_SEED, TEST_SIZE, CV_FOLDS

# Try to import XGBoost - make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available:", xgb.__version__)
except ImportError:
    print("Warning: XGBoost not available. Skipping XGBoost model.")
    XGBOOST_AVAILABLE = False

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imbalanced-learn not available. Proceeding without SMOTE.")
    SMOTE_AVAILABLE = False

MODEL_CONFIGS = {
    'logistic_regression': {
        'model': LogisticRegression(random_state=RANDOM_SEED),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'max_iter': [1000]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=RANDOM_SEED),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
    },
    'xgboost': {
        'model': xgb.XGBClassifier(random_state=RANDOM_SEED) if XGBOOST_AVAILABLE else None,
        'params': {
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7]
        }
    } if XGBOOST_AVAILABLE else None
}

def load_data():
    """Load and prepare the fraud detection dataset."""
    print("Loading and preparing data...")
    
    # Check for processed data
    if os.path.exists('data/processed/engineered_features.csv'):
        print("Using engineered features from data/processed/engineered_features.csv")
        df = pd.read_csv('data/processed/engineered_features.csv')
    elif os.path.exists('data/processed/selected_features.csv'):
        print("Using selected features from data/processed/selected_features.csv")
        df = pd.read_csv('data/processed/selected_features.csv')
    else:
        # Use the common utility to load raw data
        df = load_creditcard_data()
        if df is None:
            print("No data found. Please run feature engineering first.")
            return None, None, None, None, None
        print("Using raw data from creditcard.csv")
    
    # Apply mappings for gender and state
    df = apply_mappings(df)
    
    # Ensure we have an 'is_fraud' column
    if 'is_fraud' not in df.columns:
        print("Error: 'is_fraud' column not found. Cannot train models.")
        return None, None, None, None, None
    
    # Drop non-numeric columns for training
    training_df = df.select_dtypes(include=['number'])
    
    # Split data into features and target
    X = training_df.drop('is_fraud', axis=1)
    y = training_df['is_fraud']
    
    # Save column names for feature importance
    feature_names = X.columns.tolist()
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    
    return X_train, X_test, y_train, y_test, feature_names

def train_model(X_train, y_train, model_name):
    """Train a single model with cross-validation and hyperparameter tuning."""
    if model_name not in MODEL_CONFIGS or MODEL_CONFIGS[model_name] is None:
        print(f"Model {model_name} not available")
        return None
        
    config = MODEL_CONFIGS[model_name]
    search = RandomizedSearchCV(
        config['model'], 
        config['params'],
        n_iter=5, 
        cv=CV_FOLDS,
        scoring='f1',
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    
    try:
        search.fit(X_train, y_train)
        model = search.best_estimator_
        
        # Save model and parameters
        joblib.dump(model, f'models/{model_name}.pkl')
        with open(f'models/{model_name}_params.json', 'w') as f:
            json.dump(search.best_params_, f, indent=2)
            
        print(f"{model_name} best score: {search.best_score_:.4f}")
        return model
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return None

def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model with grid search for hyperparameters."""
    # Define hyperparameters for grid search - minimal set to avoid excessive computation
    param_grid = {
        'C': [0.1, 1.0],  # Drastically reduced parameter space
        'class_weight': ['balanced'],  # Only use balanced weights for imbalanced data
        'penalty': ['l2'],  # Only use L2 regularization
        'solver': ['liblinear']  # Compatible with L2
    }
    
    # Create the model
    logreg = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train with cross-validation
    return train_model(X_train, y_train, 'logistic_regression')

def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier with hyperparameter tuning."""
    # Get dataset dimensions to adjust model complexity
    n_samples, n_features = X_train.shape
    
    # Minimal parameter grid to prevent excessive computation
    param_grid = {
        'n_estimators': [100],  # Single value to reduce computation
        'max_depth': [10, None],  # Limited depth options
        'min_samples_split': [2],  # Default value
        'class_weight': ['balanced']  # Only use balanced weights
    }
    
    # Create the model with stronger regularization to prevent overfitting
    rf = RandomForestClassifier(
        random_state=42, 
        n_jobs=-1,  # Use parallel processing
        max_features='sqrt',  # Restrict features to prevent overfitting
        min_samples_leaf=4,   # Require more samples in leaf nodes
        oob_score=True  # Use out-of-bag scoring to detect overfitting
    )
    
    # Train with cross-validation
    return train_model(X_train, y_train, 'random_forest')

def train_xgboost(X_train, y_train):
    """Train XGBoost model with hyperparameter tuning."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available. Skipping.")
        return None
    
    try:
        # Create the model with strong regularization to prevent overfitting
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', 
            random_state=42, 
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,  # Lower depth to prevent overfitting
            subsample=0.8,  # Use subsampling to reduce overfitting
            colsample_bytree=0.8,  # Use feature subsampling to reduce overfitting
            scale_pos_weight=1.0,  # Adjust based on class imbalance
            eval_metric='logloss'  # Monitor logloss during training
        )
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [50, 100, 200],
            'reg_alpha': [0.1, 0.5, 1.0],  # L1 regularization
            'reg_lambda': [0.1, 1.0, 10.0]  # L2 regularization
        }
        
        # Use a smaller grid for faster training
        fast_param_grid = {
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1],
            'reg_lambda': [1.0, 10.0]
        }
        
        # Get initial CV score
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='f1')
        print(f"XGBoost initial CV f1 score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Check for potential overfitting
        if cv_scores.mean() > 0.99:
            print("WARNING: Very high CV scores (%.4f) suggest potential data leakage or overfitting" % cv_scores.mean())
            print("Continuing with more regularization to prevent overfitting")
            
            # Increase regularization
            xgb_model.set_params(
                reg_alpha=1.0,
                reg_lambda=10.0,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.7,
                colsample_bytree=0.7
            )
            
            # Use a smaller parameter grid focused on regularization
            fast_param_grid = {
                'reg_alpha': [1.0, 5.0, 10.0],
                'reg_lambda': [10.0, 50.0, 100.0]
            }
        
        print("Using RandomizedSearchCV with 5 iterations to reduce computation")
        grid_search = RandomizedSearchCV(
            xgb_model, 
            fast_param_grid, 
            n_iter=5,
            cv=5, 
            scoring='f1', 
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        # Train the model with a subset of data if data is large
        if len(X_train) > 10000:
            print("Using a subset of data for hyperparameter tuning")
            subset_indices = np.random.choice(len(X_train), 10000, replace=False)
            X_subset, y_subset = X_train[subset_indices], y_train[subset_indices]
            grid_search.fit(X_subset, y_subset)
        else:
            grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"XGBoost best parameters: {grid_search.best_params_}")
        print(f"XGBoost CV score: {grid_search.best_score_:.4f}")
        
        # Train on full data with best parameters
        if len(X_train) > 10000:
            print("Training final model on full dataset")
            best_model.fit(X_train, y_train)
        
        return best_model
    
    except Exception as e:
        print(f"Error training XGBoost: {str(e)}")
        return None

def evaluate_models(models, X_test, y_test):
    """Evaluate all trained models on the test set."""
    print("\nModel Evaluation:")
    print("-" * 50)
    
    results = {}
    
    # Skip evaluation if test set is empty
    if X_test.size == 0 or y_test.size == 0:
        print("Warning: Empty test set. Skipping evaluation.")
        return {}
    
    # Create a common figure for ROC curves
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if model is None:
            print(f"\n{name} model was not trained. Skipping evaluation.")
            continue
            
        try:
            # Get predictions and probabilities if available
            if name == 'Neural Network':
                y_pred_prob = model.predict(X_test)
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            else:
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_pred_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_prob = y_pred  # Use predictions if probabilities not available
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            
            # Check if y_test has at least two unique classes for ROC AUC
            if len(np.unique(y_test)) > 1:
                from sklearn.metrics import roc_curve, precision_recall_curve
                
                # Get ROC curve data
                fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
                roc_auc = roc_auc_score(y_test, y_pred_prob)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
                
                # Get precision-recall curve (better for imbalanced datasets)
                precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
                pr_auc = np.trapz(precision, recall)  # Area under PR curve
                
                # Create separate PR curve figure
                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'{name} Precision-Recall Curve (AUC = {pr_auc:.3f})')
                plt.savefig(f'models/figures/{name.lower().replace(" ", "_")}_pr_curve.png')
                plt.close()
            else:
                roc_auc = float('nan')
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': y_pred
            }
            
            # Print results
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.4f}")
            if not np.isnan(roc_auc):
                print(f"ROC AUC: {roc_auc:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            # Create directories for figures if they don't exist
            os.makedirs('models/figures', exist_ok=True)
            plt.savefig(f'models/figures/{name.lower().replace(" ", "_")}_confusion_matrix.png')
            plt.close()
        except Exception as e:
            print(f"Error evaluating {name} model: {e}")
    
    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig('models/figures/all_models_roc_curves.png')
    plt.close()
    
    return results

def create_ensemble_model(models, X_train, y_train, X_test, y_test):
    """Create a simple voting ensemble model."""
    print("\nCreating ensemble model...")
    
    # Filter out None models
    valid_models = {name: model for name, model in models.items() if model is not None}
    
    if not valid_models:
        print("No valid models available for ensemble. Skipping ensemble creation.")
        return None
    
    try:
        # Get predictions from each model
        predictions = []
        for name, model in valid_models.items():
            try:
                if name == 'Neural Network':
                    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
                else:
                    y_pred = model.predict(X_test)
                predictions.append(y_pred)
            except Exception as e:
                print(f"Error getting predictions from {name} model: {e}")
        
        if not predictions:
            print("No valid predictions for ensemble. Skipping ensemble creation.")
            return None
        
        # Ensemble by majority voting
        ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(ensemble_pred == y_test)
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, ensemble_pred)
        else:
            roc_auc = float('nan')
        
        # Print results
        print("\nEnsemble Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        if not np.isnan(roc_auc):
            print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, ensemble_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, ensemble_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix - Ensemble Model')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('models/figures/ensemble_confusion_matrix.png')
        plt.close()
        
        # Save ensemble predictions
        np.save('models/ensemble_predictions.npy', ensemble_pred)
        
        return ensemble_pred
    except Exception as e:
        print(f"Error creating ensemble model: {e}")
        return None

def plot_feature_importance(model_path, feature_names=None):
    """Plot feature importance from a trained model.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    feature_names : list, optional
        List of feature names for better labeling
    """
    try:
        # Load the model
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
            
        model = joblib.load(model_path)
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print(f"Model does not have feature importance information")
            return
            
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # If feature_names not provided, use indices
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Create directory for figures if it doesn't exist
        os.makedirs('models/figures', exist_ok=True)
        
        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(min(20, len(importances))), 
                importances[indices][:20],
                align='center')
        plt.xticks(range(min(20, len(importances))), 
                  [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                   for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        plt.savefig('models/figures/feature_importance.png')
        plt.close()
        
        print("Feature importance plot saved to 'models/figures/feature_importance.png'")
        
        # Also print top 10 features in console
        print("\nTop 10 most important features:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
            print(f"{i+1}. {name}: {importances[idx]:.4f}")
            
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")

def run_training_pipeline(X_train, X_test, y_train, y_test, feature_names=None):
    """Run the full model training pipeline."""
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Train models
    models = {}
    for model_name in MODEL_CONFIGS:
        print(f"\nTraining {model_name}...")
        model = train_model(X_train_scaled, y_train, model_name)
        if model is not None:
            models[model_name] = model
            
            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            print(f"\n{model_name} Test Set Performance:")
            print(classification_report(y_test, y_pred))
    
    # Save feature names
    feature_names = (feature_names if feature_names else 
                    list(X_train.columns) if hasattr(X_train, 'columns') else 
                    list(map(str, range(X_train.shape[1]))))
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating models on test set...")
    
    print("\nModel Evaluation:")
    print("-"*50)
    
    # Evaluate each model
    results = evaluate_models(models, X_test_scaled, y_test)
    
    # Only create ensemble if we have at least two models
    if len(models) >= 2:
        print("\n" + "="*50)
        print("Creating voting ensemble model...")
        
        ensemble = create_ensemble_model(models, X_train_scaled, y_train, X_test_scaled, y_test)
        if ensemble is not None:
            models['ensemble'] = ensemble
    
    return models

def main():
    """Main function to run model training."""
    print_header("SecurePay Fraud Detection - Model Training")
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    if X_train is None:
        print("Error preparing data. Exiting.")
        return
    
    # Train and evaluate models
    run_training_pipeline(X_train, X_test, y_train, y_test, feature_names)
    
    # Plot feature importance
    try:
        plot_feature_importance('models/random_forest.pkl', feature_names)
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        
    print_header("Model Training Complete")
    
if __name__ == "__main__":
    main() 