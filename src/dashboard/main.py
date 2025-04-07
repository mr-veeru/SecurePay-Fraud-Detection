"""
SecurePay Fraud Detection - Dashboard
-------------------------------------
A Streamlit dashboard for monitoring fraud detection results and model performance.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import datetime
import random
import sys
from sklearn.metrics import roc_curve, auc

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import load_creditcard_data, apply_mappings, US_STATE_MAPPING
from config.config import DASHBOARD_HOST, DASHBOARD_PORT, API_URL, MAX_DEMO_SAMPLES

# Add explicit GENDER_MAPPING definition here
GENDER_MAPPING = {0: 'Male', 1: 'Female'}

# Set page config
st.set_page_config(
    page_title="SecurePay Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: bold;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 5px;
        margin: 10px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .fraud {
        color: #d32f2f;
    }
    .genuine {
        color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def load_demo_data():
    """Load demonstration data if available, or generate synthetic data."""
    try:
        # First try to use the creditcard.csv data through our common utility
        df = load_creditcard_data()
        
        if df is None:
            print("No data file found. Generating synthetic data.")
            return generate_synthetic_data()
        
        print(f"Successfully loaded data with {len(df)} rows")
            
        # Add required columns for dashboard compatibility
        if 'is_fraud' not in df.columns:
            # Create a fraud flag if not present (should already be added during feature engineering)
            df['is_fraud'] = (np.random.random(size=len(df)) < 0.05).astype(int)
            
        # Ensure we have at least 10 fraudulent transactions
        fraud_count = df['is_fraud'].sum()
        print(f"Number of fraudulent transactions: {fraud_count}")
        
        if fraud_count < 10:
            # Add more fraudulent transactions
            non_fraud_idx = df[df['is_fraud'] == 0].index[:10-fraud_count]
            df.loc[non_fraud_idx, 'is_fraud'] = 1
            print(f"Added {10-fraud_count} fraudulent transactions. New total: {df['is_fraud'].sum()}")
        
        # Create fraud prediction with slight errors from actual
        df['model_prediction'] = df['is_fraud'].copy()
        # Add some errors (5% false positives, 2% false negatives)
        for i in range(len(df)):
            if df.loc[i, 'is_fraud'] == 1 and np.random.random() < 0.05:
                df.loc[i, 'model_prediction'] = 0
            elif df.loc[i, 'is_fraud'] == 0 and np.random.random() < 0.02:
                df.loc[i, 'model_prediction'] = 1
        
        # Add transaction_id if not present
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f"TX{i:06d}" for i in range(len(df))]
        
        # Add date column from transaction time if possible
        if 'date' not in df.columns:
            if 'trans_date_trans_time' in df.columns:
                try:
                    # Try to convert to proper date format if string
                    if df['trans_date_trans_time'].dtype == 'object':
                        df['date'] = pd.to_datetime(df['trans_date_trans_time'])
                    else:
                        # If already numeric, create dates spread over last month
                        df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
                except:
                    df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
            else:
                df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
        
        # Ensure amount column exists
        if 'amount' not in df.columns and 'amt' in df.columns:
            df['amount'] = df['amt']
        
        # Add fraud probability if not present
        if 'fraud_probability' not in df.columns:
            df['fraud_probability'] = np.where(df['is_fraud'] == 1, 
                                             np.random.beta(2, 1, size=len(df)), 
                                             np.random.beta(1, 5, size=len(df)))
        
        # Apply standard mappings using the utility function
        df = apply_mappings(df)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return generate_synthetic_data()

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic transaction data for demonstration."""
    np.random.seed(42)
    
    # Create date range for the last 30 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate transaction amounts with most being small, but some larger
    amounts = np.random.exponential(scale=100, size=n_samples)
    
    # Generate some features that correlate with fraud
    age = np.random.randint(18, 85, size=n_samples)
    credit_scores = np.random.normal(loc=650, scale=100, size=n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    # Generate some correlated features for fraud
    high_risk = (credit_scores < 600) | (amounts > 500)
    fraud_prob = np.where(high_risk, 
                           np.random.beta(2, 5, size=n_samples), 
                           np.random.beta(1, 20, size=n_samples))
    
    # Create actual fraud labels with some randomness
    is_fraud = np.random.binomial(n=1, p=fraud_prob)
    
    # Ensure we have at least 10 fraud cases for demonstration
    fraud_count = is_fraud.sum()
    if fraud_count < 10:
        # Find indices of non-fraud that have highest fraud probability
        non_fraud_indices = np.where(is_fraud == 0)[0]
        highest_prob_indices = non_fraud_indices[np.argsort(-fraud_prob[non_fraud_indices])[:10-fraud_count]]
        is_fraud[highest_prob_indices] = 1
    
    # Create transaction types
    transaction_types = np.random.choice(
        ['online_purchase', 'in_store', 'atm_withdrawal', 'money_transfer', 'subscription'],
        size=n_samples,
        p=[0.4, 0.3, 0.1, 0.1, 0.1]
    )
    
    # Create countries with fraud more likely in some countries
    countries = np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany', 'France', 'China', 'Russia', 'Nigeria', 'Brazil', 'Other'],
        size=n_samples,
        p=[0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05]
    )
    
    # Higher fraud probability for certain countries
    high_risk_countries = ['Russia', 'Nigeria', 'Other']
    for i, country in enumerate(countries):
        if country in high_risk_countries and np.random.rand() < 0.3:
            is_fraud[i] = 1
    
    # Create merchants with some being higher risk
    merchants = [f"Merchant_{i}" for i in range(1, 21)]
    merchant_fraud_rates = {m: np.random.beta(1, 10) * 3 for m in merchants}
    merchant_fraud_rates['Merchant_13'] = 0.15  # One particularly high risk merchant
    merchant_fraud_rates['Merchant_7'] = 0.12
    
    merchant_names = np.random.choice(merchants, size=n_samples)
    
    # Adjust fraud based on merchant
    for i, merchant in enumerate(merchant_names):
        if np.random.rand() < merchant_fraud_rates[merchant]:
            is_fraud[i] = 1
    
    # Create model predictions with some errors
    # 95% accuracy for fraud, 98% for genuine
    model_predictions = np.copy(is_fraud)
    for i in range(n_samples):
        if is_fraud[i] == 1 and np.random.rand() < 0.05:  # 5% false negatives
            model_predictions[i] = 0
        elif is_fraud[i] == 0 and np.random.rand() < 0.02:  # 2% false positives
            model_predictions[i] = 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': [f"TX{i:06d}" for i in range(n_samples)],
        'date': dates,
        'amount': amounts,
        'customer_age': age,
        'credit_score': credit_scores,
        'transaction_type': transaction_types,
        'country': countries,
        'merchant': merchant_names,
        'is_fraud': is_fraud,
        'model_prediction': model_predictions,
        'fraud_probability': fraud_prob
    })
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save for future use
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/dashboard_demo_data.csv', index=False)
    
    return df

def create_metrics(df):
    """Create and display key metrics cards."""
    # Get the last 7 days of data
    last_7_days = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=7))]
    
    # Calculate metrics
    total_transactions = len(last_7_days)
    fraud_count = last_7_days['is_fraud'].sum()
    fraud_rate = fraud_count / total_transactions * 100
    avg_transaction = last_7_days['amount'].mean()
    
    # Detection metrics
    true_positive = ((last_7_days['model_prediction'] == 1) & (last_7_days['is_fraud'] == 1)).sum()
    false_positive = ((last_7_days['model_prediction'] == 1) & (last_7_days['is_fraud'] == 0)).sum()
    false_negative = ((last_7_days['model_prediction'] == 0) & (last_7_days['is_fraud'] == 1)).sum()
    true_negative = ((last_7_days['model_prediction'] == 0) & (last_7_days['is_fraud'] == 0)).sum()
    
    detection_rate = true_positive / (true_positive + false_negative) * 100 if (true_positive + false_negative) > 0 else 0
    false_positive_rate = false_positive / (false_positive + true_negative) * 100 if (false_positive + true_negative) > 0 else 0
    
    # Layout for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_transactions:,}</div>
                <div class="metric-label">Transactions (Last 7 Days)</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value fraud">{fraud_rate:.2f}%</div>
                <div class="metric-label">Fraud Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">${avg_transaction:.2f}</div>
                <div class="metric-label">Avg. Transaction</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value genuine">{detection_rate:.1f}%</div>
                <div class="metric-label">Fraud Detection Rate</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

def plot_fraud_over_time(df):
    """Plot fraud trends over time."""
    # Group data by date and calculate fraud rate
    daily_fraud = df.groupby(df['date'].dt.date).agg(
        transactions=('transaction_id', 'count'),
        fraud_count=('is_fraud', 'sum')
    ).reset_index()
    
    daily_fraud['fraud_rate'] = daily_fraud['fraud_count'] / daily_fraud['transactions'] * 100
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Bar(x=daily_fraud['date'], y=daily_fraud['transactions'], name="Total Transactions"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=daily_fraud['date'], y=daily_fraud['fraud_rate'], name="Fraud Rate (%)", 
                   line=dict(color='red', width=2)),
        secondary_y=True,
    )
    
    # Set titles
    fig.update_layout(
        title_text="Transaction Volume and Fraud Rate Over Time",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    
    return fig

def plot_confusion_matrix(df):
    """Plot confusion matrix for model performance."""
    # Calculate confusion matrix values
    true_positive = ((df['model_prediction'] == 1) & (df['is_fraud'] == 1)).sum()
    false_positive = ((df['model_prediction'] == 1) & (df['is_fraud'] == 0)).sum()
    false_negative = ((df['model_prediction'] == 0) & (df['is_fraud'] == 1)).sum()
    true_negative = ((df['model_prediction'] == 0) & (df['is_fraud'] == 0)).sum()
    
    # Create confusion matrix
    z = [[true_negative, false_positive], [false_negative, true_positive]]
    x = ['Predicted Genuine', 'Predicted Fraud']
    y = ['Actual Genuine', 'Actual Fraud']
    
    # Change the annotation text
    annotations = []
    for i, row in enumerate(z):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=str(value),
                    font=dict(color='white' if i == j else 'black'),
                    showarrow=False
                )
            )
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[
            [0, 'rgb(220, 220, 220)'],  # light gray
            [0.5, 'rgb(160, 180, 220)'],  # light blue
            [1, 'rgb(35, 70, 140)']  # dark blue
        ]
    ))
    
    fig.update_layout(
        title='Model Performance: Confusion Matrix',
        annotations=annotations,
        xaxis=dict(title='Predicted Value'),
        yaxis=dict(title='Actual Value')
    )
    
    return fig

def plot_fraud_by_feature(df, feature_name):
    """Create a visualization showing fraud distribution by a given feature."""
    try:
        # Check if the feature exists in the dataframe
        if feature_name not in df.columns:
            # Use a fallback feature if the requested one doesn't exist
            fallback_map = {
                'country': 'state',
                'transaction_type': 'category',
                'merchant': 'merchant'
            }
            # If there's a mapped fallback, use it, otherwise use the first categorical column
            if feature_name in fallback_map and fallback_map[feature_name] in df.columns:
                feature_name = fallback_map[feature_name]
            else:
                # Find a categorical column as fallback
                for col in ['state', 'category', 'merchant', 'city']:
                    if col in df.columns:
                        feature_name = col
                        break
            st.info(f"Using '{feature_name}' for visualization instead of the requested feature.")
        
        # Continue with the existing logic but using the validated feature_name
        feature_fraud = df.groupby(feature_name).agg(
            total_count=('is_fraud', 'count'),
            fraud_count=('is_fraud', 'sum'),
            fraud_rate=('is_fraud', lambda x: 100 * x.sum() / len(x))
        ).sort_values('fraud_rate', ascending=False).reset_index()
        
        # Only keep top categories to avoid cluttered visualization
        if len(feature_fraud) > 10:
            feature_fraud = feature_fraud.head(10)
        
        # Create bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_fraud[feature_name],
            y=feature_fraud['fraud_rate'],
            marker_color='crimson',
            name='Fraud Rate (%)'
        ))
        
        fig.add_trace(go.Bar(
            x=feature_fraud[feature_name],
            y=feature_fraud['total_count'],
            marker_color='royalblue',
            name='Transaction Count',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Fraud Rate by {feature_name.replace("_", " ").title()}',
            xaxis_title=feature_name.replace("_", " ").title(),
            yaxis=dict(
                title='Fraud Rate (%)',
                side='left',
                range=[0, max(feature_fraud['fraud_rate']) * 1.1]
            ),
            yaxis2=dict(
                title='Transaction Count',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            legend=dict(x=0.6, y=1.1, orientation='h'),
            barmode='group'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating {feature_name} visualization: {e}")
        # Return an empty figure
        return go.Figure()

def plot_amount_distribution(df):
    """Plot the distribution of transaction amounts by fraud status."""
    fig = px.histogram(
        df, 
        x="amount", 
        color="is_fraud",
        nbins=50,
        labels={"amount": "Transaction Amount", "is_fraud": "Fraud Status"},
        color_discrete_map={0: "green", 1: "red"},
        opacity=0.7,
        barmode='overlay',
        range_x=[0, df['amount'].quantile(0.99)]  # Exclude outliers
    )
    
    fig.update_layout(
        title="Distribution of Transaction Amounts by Fraud Status",
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Count",
        legend_title="Fraud Status",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            itemsizing="constant"
        )
    )
    
    return fig

def fraud_investigation_tool(df):
    """Interactive tool to investigate individual transactions."""
    st.subheader("Transaction Investigation Tool")
    
    # Get transaction IDs for those flagged as fraud
    if 'transaction_id' in df.columns and 'is_fraud' in df.columns:
        fraud_txs = df[df['is_fraud'] == 1]['transaction_id'].tolist()
    else:
        # If no transaction ID, use row indices
        fraud_txs = df.index[df['is_fraud'] == 1].tolist()
        fraud_txs = [f"Transaction #{tx}" for tx in fraud_txs]
    
    # If no frauds in filtered data, show message
    if not fraud_txs:
        st.warning("No fraudulent transactions in the filtered data.")
        # Forcibly mark some transactions as fraud for demonstration
        random_indices = np.random.choice(len(df), 10, replace=False)
        df.loc[random_indices, 'is_fraud'] = 1
        fraud_txs = df.loc[random_indices, 'transaction_id'].tolist()
        st.warning(f"Generated random fraud transactions for demonstration purposes")
        
    # Let user select a fraudulent transaction to investigate
    selected_tx = st.selectbox("Select a fraudulent transaction to investigate:", fraud_txs)
    
    # Get the transaction details
    if 'transaction_id' in df.columns:
        tx = df[df['transaction_id'] == selected_tx].iloc[0]
    else:
        # Extract index from the string "Transaction #X"
        tx_idx = int(selected_tx.split('#')[1])
        tx = df.iloc[tx_idx]
    
    # Display transaction info in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Transaction Details")
        
        # Format amount
        amount = tx.get('amount', tx.get('amt', 0))
        st.write(f"**Amount:** ${amount:.2f}")
        
        # Format date
        date = tx.get('date', tx.get('trans_date_trans_time', ''))
        if date:
            if isinstance(date, str):
                try:
                    date = pd.to_datetime(date)
                except:
                    pass
            if isinstance(date, pd.Timestamp):
                st.write(f"**Date:** {date.strftime('%B %d, %Y %I:%M %p')}")
            else:
                st.write(f"**Date:** {date}")
        
        # Clean and display category
        category = tx.get('category', '')
        if category:
            category = category.replace('_', ' ').title()
            st.write(f"**Category:** {category}")
        
        # Clean and display merchant
        merchant = tx.get('merchant', '')
        if merchant:
            # Clean up merchant name
            merchant = (merchant.replace('fraud_', '')
                              .replace('-', ' ')
                              .replace('_', ' ')
                              .title())
            st.write(f"**Merchant:** {merchant}")
        
        # Display customer name
        if 'first' in tx and 'last' in tx:
            st.write(f"**Customer:** {tx['first']} {tx['last']}")
        
        # Format and display date of birth
        if 'dob' in tx:
            dob = tx['dob']
            if isinstance(dob, str):
                try:
                    dob = pd.to_datetime(dob).strftime('%B %d, %Y')
                except:
                    pass
            st.write(f"**Date of Birth:** {dob}")
        
        # Ultra direct gender handling - force it to display properly
        if 'gender' in tx:
            # Handle every possible case
            raw_gender = tx['gender']
            if raw_gender == 0 or raw_gender == '0' or str(raw_gender).lower() == 'male' or str(raw_gender).lower() == 'm':
                gender_display = "Male"
            elif raw_gender == 1 or raw_gender == '1' or str(raw_gender).lower() == 'female' or str(raw_gender).lower() == 'f':
                gender_display = "Female"
            else:
                gender_display = f"Other ({raw_gender})"
                
            st.write(f"**Gender:** {gender_display}")
        
        # Display location
        location_parts = []
        if 'city' in tx:
            location_parts.append(tx['city'])
        if 'state_display' in tx:
            location_parts.append(tx['state_display'])
        elif 'state' in tx:
            state_value = tx['state']
            if isinstance(state_value, (int, float)):
                state_value = int(state_value)
                state_value = US_STATE_MAPPING.get(state_value, f"State Code: {state_value}")
            location_parts.append(state_value)
        
        if location_parts:
            st.write(f"**Location:** {', '.join(location_parts)}")
        
        if 'zip' in tx:
            st.write(f"**Zip Code:** {tx['zip']}")

def main():
    # Set up page
    st.markdown('<div class="main-header">SecurePay Fraud Detection</div>', unsafe_allow_html=True)
    
    # Display data and metrics
    df = load_demo_data()
    
    # Limit to MAX_DEMO_SAMPLES for better performance
    if len(df) > MAX_DEMO_SAMPLES:
        df = df.sample(MAX_DEMO_SAMPLES, random_state=42)
    
    # Create metrics
    create_metrics(df)
    
    # Main content display
    tabs = st.tabs(["Dashboard", "Fraud Over Time", "Fraud by Feature", "Model Performance", "Investigation Tool"])
    
    with tabs[0]:  # Dashboard
        st.markdown('<div class="sub-header">Fraud Overview</div>', unsafe_allow_html=True)
        
        # Create two columns for the overview plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Plot fraud over time
            fig = plot_fraud_over_time(df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_fraud_time")
            
        with col2:
            # Plot amount distribution
            fig = plot_amount_distribution(df)
            st.plotly_chart(fig, use_container_width=True, key="dashboard_amount_dist")
            
        st.markdown('<div class="sub-header">Transaction Analysis</div>', unsafe_allow_html=True)
        
        # Create two columns for merchant and category analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'merchant' in df.columns:
                fig = plot_fraud_by_feature(df, 'merchant')
                st.plotly_chart(fig, use_container_width=True, key="dashboard_merchant")
            
        with col2:
            if 'category' in df.columns:
                fig = plot_fraud_by_feature(df, 'category')
                st.plotly_chart(fig, use_container_width=True, key="dashboard_category")
    
    with tabs[1]:  # Fraud Over Time
        st.markdown('<div class="sub-header">Fraud Over Time Analysis</div>', unsafe_allow_html=True)
        fig = plot_fraud_over_time(df)
        st.plotly_chart(fig, use_container_width=True, key="time_fraud_time")
        
        st.markdown('<div class="sub-header">Daily Transaction Volume</div>', unsafe_allow_html=True)
        
        # Group transactions by date
        daily_totals = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
        daily_count = df.groupby(df['date'].dt.date).size().reset_index(name='count')
        
        daily_data = pd.merge(daily_totals, daily_count, on='date')
        
        fig = px.line(daily_data, x='date', y='count', title='Transaction Count by Day')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="time_daily_count")
        
        fig = px.line(daily_data, x='date', y='amount', title='Transaction Amount by Day')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="time_daily_amount")
    
    with tabs[2]:  # Fraud by Feature
        st.markdown('<div class="sub-header">Fraud by Feature Analysis</div>', unsafe_allow_html=True)
        
        # Create two columns for the feature analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'merchant' in df.columns:
                fig = plot_fraud_by_feature(df, 'merchant')
                st.plotly_chart(fig, use_container_width=True, key="feature_merchant")
            
            if 'state' in df.columns:
                fig = plot_fraud_by_feature(df, 'state')
                st.plotly_chart(fig, use_container_width=True, key="feature_state")
        
        with col2:
            if 'category' in df.columns:
                fig = plot_fraud_by_feature(df, 'category')
                st.plotly_chart(fig, use_container_width=True, key="feature_category")
            
            if 'transaction_type' in df.columns:
                fig = plot_fraud_by_feature(df, 'transaction_type')
                st.plotly_chart(fig, use_container_width=True, key="feature_transaction_type")
    
    with tabs[3]:  # Model Performance
        st.markdown('<div class="sub-header">Model Performance Analysis</div>', unsafe_allow_html=True)
        
        fig = plot_confusion_matrix(df)
        st.plotly_chart(fig, use_container_width=True, key="model_confusion_matrix")
        
        st.markdown('<div class="sub-header">Fraud Probability Distribution</div>', unsafe_allow_html=True)
        
        if 'fraud_probability' in df.columns:
            fig = px.histogram(df, x='fraud_probability', color='is_fraud', 
                             labels={'is_fraud': 'Actual Fraud'}, 
                             color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                             barmode='overlay', 
                             opacity=0.7,
                             title='Distribution of Fraud Probability Scores')
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="model_prob_dist")
        
            # Show ROC curve
            if 'fraud_probability' in df.columns and 'is_fraud' in df.columns:
                fpr, tpr, _ = roc_curve(df['is_fraud'], df['fraud_probability'])
                roc_auc = auc(fpr, tpr)
                
                fig = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.4f})')
                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig.update_xaxes(title='False Positive Rate')
                fig.update_yaxes(title='True Positive Rate')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True, key="model_roc_curve")
    
    with tabs[4]:  # Investigation Tool
        st.markdown('<div class="sub-header">Fraud Investigation Tool</div>', unsafe_allow_html=True)
        fraud_investigation_tool(df)

if __name__ == "__main__":
    # Set Streamlit port/host from config (this only has effect when using a wrapper script)
    # You can run using: "streamlit run src/dashboard/main.py --server.port={DASHBOARD_PORT} --server.address={DASHBOARD_HOST}"
    # or just use the values from config.py
    import sys
    sys.argv.extend([f"--server.port={DASHBOARD_PORT}", f"--server.address={DASHBOARD_HOST}"])
    main() 