#!/usr/bin/env python3
"""
Sample Data Generator for Fraud Detection System
Generates realistic synthetic credit card transaction data for testing and development.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random


def generate_synthetic_transactions(n_samples=10000):
    """
    Generate synthetic credit card transaction data.
    
    Args:
        n_samples (int): Number of transactions to generate
        
    Returns:
        pd.DataFrame: Synthetic transaction data
    """
    print(f"Generating {n_samples} synthetic transactions...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = [f"TX{i:06d}" for i in range(n_samples)]
    
    # Generate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate transaction amounts (exponential distribution for realistic amounts)
    amounts = np.random.exponential(scale=100, size=n_samples)
    amounts = np.clip(amounts, 1.0, 5000.0)  # Limit between $1 and $5000
    
    # Generate customer demographics
    ages = np.random.normal(loc=45, scale=15, size=n_samples)
    ages = np.clip(ages, 18, 85).astype(int)
    
    credit_scores = np.random.normal(loc=650, scale=100, size=n_samples)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # Generate geographic coordinates (US-based)
    latitudes = np.random.normal(loc=39.8283, scale=10, size=n_samples)  # US center
    longitudes = np.random.normal(loc=-98.5795, scale=15, size=n_samples)
    
    # Generate merchant coordinates (near customer locations)
    merchant_lats = latitudes + np.random.normal(0, 0.1, size=n_samples)
    merchant_longs = longitudes + np.random.normal(0, 0.1, size=n_samples)
    
    # Generate transaction types
    transaction_types = np.random.choice(
        ['online_purchase', 'in_store', 'atm_withdrawal', 'money_transfer'],
        size=n_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Generate merchant categories
    merchant_categories = np.random.choice(
        ['retail', 'food', 'gas', 'entertainment', 'travel', 'utilities'],
        size=n_samples,
        p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05]
    )
    
    # Generate time-based features
    hours = timestamps.hour
    days_of_week = timestamps.dayofweek
    
    # Generate distance features
    distances = np.sqrt(
        (latitudes - merchant_lats)**2 + 
        (longitudes - merchant_longs)**2
    ) * 69  # Convert to miles (roughly)
    
    # Generate synthetic fraud labels based on risk factors
    fraud_probability = np.zeros(n_samples)
    
    # High-risk factors increase fraud probability
    fraud_probability += (amounts > 1000) * 0.3  # High amounts
    fraud_probability += (credit_scores < 600) * 0.2  # Low credit scores
    fraud_probability += (distances > 100) * 0.4  # Large distances
    fraud_probability += (hours < 6) * 0.1  # Late night transactions
    fraud_probability += (transaction_types == 'money_transfer') * 0.2  # Money transfers
    
    # Add some randomness
    fraud_probability += np.random.normal(0, 0.05, size=n_samples)
    fraud_probability = np.clip(fraud_probability, 0, 1)
    
    # Generate fraud labels
    is_fraud = (np.random.random(size=n_samples) < fraud_probability).astype(int)
    
    # Ensure reasonable fraud rate (~2-5%)
    fraud_rate = is_fraud.mean()
    if fraud_rate > 0.05:
        # Reduce fraud rate by randomly flipping some fraud cases
        fraud_indices = np.where(is_fraud == 1)[0]
        flip_count = int(len(fraud_indices) * (fraud_rate - 0.03) / fraud_rate)
        if flip_count > 0:
            flip_indices = np.random.choice(fraud_indices, size=flip_count, replace=False)
            is_fraud[flip_indices] = 0
    
    # Create the dataframe
    data = {
        'transaction_id': transaction_ids,
        'timestamp': timestamps,
        'amount': amounts,
        'customer_age': ages,
        'credit_score': credit_scores,
        'customer_lat': latitudes,
        'customer_long': longitudes,
        'merchant_lat': merchant_lats,
        'merchant_long': merchant_longs,
        'transaction_type': transaction_types,
        'merchant_category': merchant_categories,
        'hour_of_day': hours,
        'day_of_week': days_of_week,
        'distance_from_customer': distances,
        'is_fraud': is_fraud
    }
    
    df = pd.DataFrame(data)
    
    # Add some derived features
    df['log_amount'] = np.log1p(df['amount'])
    df['amount_bin_low'] = (df['amount'] < 50).astype(int)
    df['amount_bin_medium'] = ((df['amount'] >= 50) & (df['amount'] < 200)).astype(int)
    df['amount_bin_high'] = (df['amount'] >= 200).astype(int)
    
    # Add risk score
    df['risk_score'] = (
        (df['amount'] > 1000) * 2 +
        (df['credit_score'] < 600) * 1.5 +
        (df['distance_from_customer'] > 100) * 2.5 +
        (df['hour_of_day'] < 6) * 1 +
        (df['transaction_type'] == 'money_transfer') * 1.5
    )
    
    print(f"✅ Generated {len(df)} transactions with {df['is_fraud'].sum()} fraud cases")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"   Amount range: ${df['amount'].min():.2f} - ${df['amount'].max():.2f}")
    
    return df


def save_data(df, output_path='data/creditcard.csv'):
    """
    Save the generated data to CSV file.
    
    Args:
        df (pd.DataFrame): Data to save
        output_path (str): Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to {output_path}")
    
    # Display sample
    print("\n📊 Sample of generated data:")
    print(df.head())
    
    # Display fraud statistics
    print(f"\n📈 Fraud Statistics:")
    print(f"   Total transactions: {len(df)}")
    print(f"   Fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"   Genuine transactions: {(df['is_fraud'] == 0).sum()}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.2%}")


def main():
    """Main function to generate sample data."""
    print("🛡️  FRAUD DETECTION - SAMPLE DATA GENERATOR")
    print("=" * 50)
    
    # Check if data already exists
    if os.path.exists('data/creditcard.csv'):
        print("⚠️  Data file already exists at data/creditcard.csv")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ Data generation cancelled.")
            return
    
    # Generate data
    try:
        df = generate_synthetic_transactions(n_samples=10000)
        save_data(df)
        
        print("\n🎉 Sample data generation completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run feature engineering: python feature_engineering.py")
        print("   2. Train models: python train_models.py")
        print("   3. Start the system: python main.py")
        
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        print("Please check your Python environment and dependencies.")


if __name__ == "__main__":
    main()
