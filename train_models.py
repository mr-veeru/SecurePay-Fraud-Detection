#!/usr/bin/env python3
"""
Simple Model Training Script
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def load_data():
    """Load and prepare the fraud detection dataset."""
    print("Loading and preparing data...")

    # Try to load processed data first
    if os.path.exists('data/processed/selected_features.csv'):
        print("Using processed data from data/processed/selected_features.csv")
        df = pd.read_csv('data/processed/selected_features.csv')
    else:
        print("No processed data found. Please run feature engineering first.")
        return None, None, None, None, None

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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(
        f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

    return X_train, X_test, y_train, y_test, feature_names


def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    print("Training Logistic Regression...")

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'models/logistic_regression.pkl')
    print("Logistic Regression model saved")

    return model


def train_random_forest(X_train, y_train):
    """Train a Random Forest classifier."""
    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'models/random_forest.pkl')
    print("Random Forest model saved")

    return model


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model on the test set."""
    print(f"\nEvaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Create directories for figures if they don't exist
    os.makedirs('models/figures', exist_ok=True)
    plt.savefig(
        f'models/figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

    return accuracy


def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance from a trained model."""
    try:
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not have feature importance information")
            return

        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Create directory for figures if it doesn't exist
        os.makedirs('models/figures', exist_ok=True)

        # Plot the feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} Feature Importances')
        plt.bar(range(min(20, len(importances))),
                importances[indices][:20],
                align='center')
        plt.xticks(range(min(20, len(importances))),
                   [feature_names[i] if i < len(feature_names) else f"Feature {i}"
                   for i in indices[:20]],
                   rotation=90)
        plt.tight_layout()
        plt.savefig(
            f'models/figures/{model_name.lower().replace(" ", "_")}_feature_importance.png')
        plt.close()

        print(f"{model_name} feature importance plot saved")

        # Print top 10 features
        print(f"\nTop 10 most important features for {model_name}:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            name = feature_names[idx] if idx < len(
                feature_names) else f"Feature {idx}"
            print(f"{i+1}. {name}: {importances[idx]:.4f}")

    except Exception as e:
        print(f"Error creating feature importance plot for {model_name}: {e}")


def main():
    """Main function to run model training."""
    print("Fraud Detection - Model Training")
    print("=" * 40)

    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    if X_train is None:
        print("Error preparing data. Exiting.")
        return

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for inference
    joblib.dump(scaler, 'models/scaler.pkl')

    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Train models
    models = {}

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    models['Logistic Regression'] = lr_model

    # Train Random Forest
    rf_model = train_random_forest(X_train_scaled, y_train)
    models['Random Forest'] = rf_model

    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating models on test set...")

    results = {}
    for name, model in models.items():
        accuracy = evaluate_model(model, X_test_scaled, y_test, name)
        results[name] = accuracy

        # Plot feature importance for Random Forest
        if name == 'Random Forest':
            plot_feature_importance(model, feature_names, name)

    # Print summary
    print("\n" + "="*50)
    print("Model Performance Summary:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.4f}")

    print("\nModel training completed!")


if __name__ == "__main__":
    main()
