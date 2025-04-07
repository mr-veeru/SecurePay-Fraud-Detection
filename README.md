# SecurePay Fraud Detection System

An end-to-end system for detecting fraudulent transactions in financial data.

## Project Overview

SecurePay Fraud Detection is a machine learning system that analyzes transaction data to identify potentially fraudulent activities. The system includes:

- Feature engineering pipeline
- Fraud detection model training 
- REST API for real-time predictions
- Interactive dashboard for monitoring and investigation

## Project Structure

```
SecurePay-Fraud-Detection/
├── data/                      # Data directory
│   ├── creditcard.csv         # Raw transaction data
│   └── processed/             # Processed and feature-engineered data
├── models/                    # Trained models and artifacts
├── logs/                      # Application logs
├── src/                       # Source code
│   ├── api/                   # REST API code
│   ├── config/                # Configuration settings
│   ├── dashboard/             # Streamlit dashboard
│   ├── features/              # Feature engineering
│   ├── models/                # Model training and evaluation
│   └── utils/                 # Shared utilities
├── requirements.txt           # Python dependencies
└── setup.py                   # Project installation script
```

## Key Features

- **Data Quality Checks**: Automatic detection of missing values, outliers, and potential data leakage issues
- **Feature Engineering**: Transforms raw transaction data into ML-ready features
- **Multi-Model Approach**: Supports Random Forest, XGBoost, and Logistic Regression 
- **Ensemble Model**: Combines multiple algorithms for improved performance
- **Real-time API**: REST API for fraud prediction on new transactions
- **Interactive Dashboard**: Visual monitoring of fraud patterns and transaction investigation
- **Centralized Configuration**: Easy customization via configuration files

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/SecurePay-Fraud-Detection.git
cd SecurePay-Fraud-Detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your transaction data in CSV format in the `data/` directory. The file should be named `creditcard.csv`.

### 2. Feature Engineering

```bash
python src/features/feature_engineering.py
```

### 3. Model Training

```bash
python src/models/train_models.py
```

### 4. Starting the API

```bash
python src/api/app.py
```

### 5. Running the Dashboard

```bash
streamlit run src/dashboard/main.py
```

The dashboard will be available at http://localhost:8501

## API Documentation

The API provides endpoints for fraud detection:

### Predict Endpoint

```
POST /api/predict
```

Example request:
```json
{
  "transaction_id": "TX123456",
  "amount": 150.75,
  "merchant": "Online Store",
  "customer_id": "C789012",
  "date": "2023-04-07T10:15:30",
  "card_type": "credit",
  "zip_code": "10001"
}
```

Example response:
```json
{
  "transaction_id": "TX123456",
  "is_fraud": 0,
  "fraud_probability": 0.12,
  "risk_factors": ["amount_above_average"],
  "model_used": "ensemble"
}
```

## Configuration

Configuration settings can be modified in `src/config/config.py` or by setting environment variables.

Key settings include:
- `API_HOST` and `API_PORT`: API server settings
- `DASHBOARD_HOST` and `DASHBOARD_PORT`: Dashboard server settings
- `RANDOM_SEED`: Seed for reproducibility
- `BALANCE_METHOD`: Method for handling class imbalance (SMOTE, ADASYN, random)

## Performance

The current model achieves:
- 99.9% Accuracy
- 92.1% Recall for fraud cases
- 95.3% Precision for fraud cases
- F1-Score: 0.937

## License

This project is licensed under the MIT License - see the LICENSE file for details. 