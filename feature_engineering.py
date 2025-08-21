#!/usr/bin/env python3
"""
Simple Feature Engineering Script
"""

import numpy as np
import pandas as pd
import os


def load_raw_data():
    """Load the raw fraud detection dataset."""
    print("Loading raw data...")

    # Try to locate the data file
    for path in [
        "data/creditcard.csv",
        "../data/creditcard.csv",
        "../../data/creditcard.csv"
    ]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(
                    f"Loaded data from {path} with {len(df)} rows and {len(df.columns)} columns")
                return df
            except Exception as e:
                print(f"Error loading data from {path}: {e}")

    print("No data found. Please download the creditcard dataset and place it in data/")
    return None


def create_synthetic_fraud_labels(df):
    """Create synthetic fraud labels based on transaction characteristics."""
    df = df.copy()

    # Identify amount column
    amount_col = 'Amount' if 'Amount' in df.columns else None

    if amount_col is None:
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and df[col].min() >= 0 and df[col].mean() < df[col].std():
                amount_col = col
                print(f"Using {col} as the transaction amount column")
                break

    if amount_col is None:
        print("Warning: Could not identify transaction amount column")
        np.random.seed(42)
        df['is_fraud'] = np.random.choice(
            [0, 1], size=len(df), p=[0.995, 0.005])
        return df

    # Create is_fraud column based on heuristics
    amount_threshold = df[amount_col].quantile(0.99)

    # Initialize fraud flags
    df['is_fraud'] = 0

    # Flag high-value transactions as potentially fraudulent
    df.loc[df[amount_col] > amount_threshold, 'is_fraud'] = 1

    # Ensure a reasonable fraud rate (~0.5-2%)
    fraud_rate = df['is_fraud'].mean()
    if fraud_rate > 0.02:
        fraud_indices = df[df['is_fraud'] == 1].index
        keep_n = int(len(df) * 0.01)
        drop_indices = np.random.choice(
            fraud_indices,
            size=max(0, len(fraud_indices) - keep_n),
            replace=False
        )
        df.loc[drop_indices, 'is_fraud'] = 0

    print(
        f"Created synthetic fraud labels. Fraud rate: {df['is_fraud'].mean():.4f}")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("Handling missing values...")

    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        print(f"Found {total_missing} missing values")

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                missing_ratio = df[col].isnull().sum() / len(df)
                print(f"  Column '{col}': {missing_ratio:.2%} missing values")

                if df[col].dtype in ['int64', 'float64']:
                    # Fill numeric columns with median
                    df[col] = df[col].fillna(df[col].median())
                else:
                    # Fill categorical columns with mode
                    df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("No missing values found")

    return df


def create_additional_features(df):
    """Create additional features that might help in fraud detection."""
    print("Creating additional features...")

    df_new = df.copy()
    new_feature_count = 0

    # Transaction amount related features
    amt_col = None
    for col in ['amt', 'amount', 'Amount']:
        if col in df.columns:
            amt_col = col
            break

    if amt_col:
        # Log transform of amount to reduce skewness
        df_new['log_amount'] = np.log1p(df[amt_col])
        new_feature_count += 1

        # Create amount bins
        q33 = df[amt_col].quantile(0.33)
        q66 = df[amt_col].quantile(0.66)

        df_new['amount_bin_low'] = (df[amt_col] <= q33).astype(int)
        df_new['amount_bin_medium'] = (
            (df[amt_col] > q33) & (df[amt_col] <= q66)).astype(int)
        df_new['amount_bin_high'] = (df[amt_col] > q66).astype(int)
        new_feature_count += 3

    # Time-based features if a timestamp column exists
    time_col = None
    for col in ['trans_date_trans_time', 'datetime', 'timestamp']:
        if col in df.columns:
            time_col = col
            break

    if time_col and pd.api.types.is_numeric_dtype(df[time_col]):
        try:
            df_new['hour_of_day'] = df[time_col] % 24
            df_new['day_of_week'] = (df[time_col] // 24) % 7
            new_feature_count += 2
        except Exception as e:
            print(f"Could not create time-based features: {e}")

    # Location-based features (if available)
    if all(col in df.columns for col in ['lat', 'long']):
        df_new['distance_from_origin'] = np.sqrt(df['lat']**2 + df['long']**2)
        new_feature_count += 1

    print(f"Created {new_feature_count} new features.")
    print(f"New dataframe shape: {df_new.shape}")

    return df_new


def convert_categorical_features(df):
    """Convert categorical features to numeric."""
    print("Converting categorical features...")

    df_processed = df.copy()

    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            print(f"Converting categorical column '{col}' to numeric")
            df_processed[col] = pd.Categorical(df_processed[col]).codes

    return df_processed


def main():
    """Main function to run the feature engineering process."""
    print("Fraud Detection - Feature Engineering")
    print("=" * 40)

    # Load the data
    df = load_raw_data()
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # Handle missing values
    df = handle_missing_values(df)

    # Create synthetic fraud labels if not present
    if 'is_fraud' not in df.columns:
        df = create_synthetic_fraud_labels(df)

    # Create additional features
    df_engineered = create_additional_features(df)

    # Convert categorical features to numeric
    df_processed = convert_categorical_features(df_engineered)

    # Save the processed data
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    df_processed.to_csv('data/processed/selected_features.csv', index=False)

    print("Feature engineering process completed.")


if __name__ == "__main__":
    main()
