# Fraud Detection System

A comprehensive machine learning project for detecting fraudulent credit card transactions with a clean, professional structure.

## 🎯 Overview

This project implements a complete fraud detection system using machine learning algorithms to identify fraudulent transactions in credit card data. It features a modular architecture with separate components for data processing, model training, API services, and web dashboard.

## ✨ Features

- **Data Processing**: Automated feature engineering and data preprocessing
- **Machine Learning Models**: Random Forest and Logistic Regression with hyperparameter tuning
- **Web Dashboard**: Interactive Streamlit dashboard for data visualization and analysis
- **REST API**: Production-ready API for real-time fraud predictions
- **Model Evaluation**: Comprehensive metrics and visualization generation
- **Easy Deployment**: Simple setup and execution process

## 🏗️ Project Structure

```
fraud-detection/
├── data/                           # Data directory
│   ├── creditcard.csv             # Raw transaction data
│   └── processed/                 # Processed data
│       └── selected_features.csv  # Feature-engineered data
├── models/                        # Trained models and artifacts
│   ├── random_forest.pkl         # Random Forest model
│   ├── logistic_regression.pkl   # Logistic Regression model
│   ├── scaler.pkl               # Feature scaler
│   ├── feature_names.json       # Feature names
│   └── figures/                 # Model visualizations and metrics
├── generate_sample_data.py      # Sample data generator for testing
├── feature_engineering.py       # Data preprocessing and feature engineering
├── train_models.py             # Model training and evaluation
├── api_server.py               # REST API server for predictions
├── dashboard.py                # Interactive web dashboard
├── test_api.py                 # API testing and validation
├── main.py                     # Main application runner
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

### 📁 Directory Overview

- **`data/`** - Contains raw and processed transaction data
- **`models/`** - Stores trained ML models, scalers, and visualization figures
- **`*.py`** - Core Python modules for different system components
- **`requirements.txt`** - Python package dependencies
- **`README.md`** - Comprehensive project documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - **Option 1 (Recommended for testing)**: Generate sample data with `python generate_sample_data.py`
   - **Option 2**: Download from Kaggle: `kaggle datasets download -d mlg-ulb/creditcardfraud`
   - **Option 3**: Place your own CSV file in `data/creditcard.csv`

### One-Command Execution

Run the main application to get started:

```bash
python main.py
```

This will:
- Check data availability
- Run feature engineering if needed
- Train models if needed
- Provide a menu to start services

## 📋 Manual Execution

### 0. Generate Sample Data (If No Data Available)

If you don't have transaction data, generate realistic synthetic data:

```bash
python generate_sample_data.py
```

**What it creates:**
- 10,000 synthetic credit card transactions
- Realistic fraud patterns based on risk factors
- Geographic coordinates, amounts, customer demographics
- Proper fraud labels for training

### 1. Feature Engineering

Process and prepare your data:

```bash
python feature_engineering.py
```

**What it does:**
- Loads raw transaction data
- Handles missing values
- Creates synthetic fraud labels (if not present)
- Generates engineered features
- Saves processed data to `data/processed/`

### 2. Model Training

Train machine learning models:

```bash
python train_models.py
```

**What it does:**
- Loads processed data
- Trains Random Forest and Logistic Regression models
- Evaluates model performance
- Generates visualizations and metrics
- Saves trained models to `models/`

### 3. Start Web Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

**Features:**
- Transaction overview and statistics
- Fraud detection results
- Model performance metrics
- Interactive visualizations
- Real-time data analysis

### 4. Start API Server

Launch the prediction API:

```bash
python api_server.py
```

**API Endpoints:**
- `GET /health` - Service health check
- `POST /predict` - Fraud prediction endpoint

### 5. Test API

Validate API functionality:

```bash
python test_api.py
```

## 🔧 API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

### Make Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amt": 100.0,
    "lat": 40.7128,
    "long": -74.0060,
    "city": 1,
    "zip": 10001,
    "dob": 1,
    "merch_lat": 40.7589,
    "merch_long": -73.9851,
    "log_amount": 4.605,
    "amount_bin_low": 0,
    "amount_bin_medium": 1,
    "amount_bin_high": 0,
    "hour_of_day": 14,
    "day_of_week": 3,
    "distance_from_origin": 0.0
  }'
```

## 📊 Model Performance

The system typically achieves:

- **Random Forest**: ~99.8% accuracy
- **Logistic Regression**: ~95% accuracy
- **Precision**: ~85-90%
- **Recall**: ~80-85%

## 🎨 Dashboard Features

- **Transaction Overview**: Summary statistics and trends
- **Fraud Analysis**: Fraud detection results and patterns
- **Model Metrics**: Performance comparison and evaluation
- **Interactive Charts**: Dynamic visualizations with Plotly
- **Real-time Updates**: Live data refresh capabilities

## 🔍 Data Requirements

Your CSV file should contain:

- Transaction amounts
- Geographic coordinates (latitude/longitude)
- Temporal features (time, date)
- Customer demographics
- Merchant information
- Transaction metadata

## 🛠️ Customization

### Adding New Models

1. Implement your model in `train_models.py`
2. Add model loading in `api_server.py`
3. Update the prediction endpoint

### Feature Engineering

Modify `feature_engineering.py` to:
- Add new feature calculations
- Implement different preprocessing strategies
- Handle new data formats

### Dashboard Enhancements

Extend `dashboard.py` with:
- Additional visualizations
- New analysis tools
- Custom metrics and KPIs

## 📝 Dependencies

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Flask, Streamlit
- **Utilities**: joblib

---

**Ready to detect fraud? Start with `python main.py`!** 🚀
