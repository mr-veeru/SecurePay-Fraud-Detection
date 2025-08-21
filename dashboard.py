#!/usr/bin/env python3
"""
Simple Fraud Detection Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os

# Set page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)


def load_demo_data():
    """Load demonstration data if available, or generate synthetic data."""
    try:
        # Try to load processed data
        if os.path.exists('data/processed/selected_features.csv'):
            df = pd.read_csv('data/processed/selected_features.csv')
            print(f"Successfully loaded data with {len(df)} rows")
        else:
            print("No processed data found. Generating synthetic data.")
            return generate_synthetic_data()

        # Add required columns for dashboard compatibility
        if 'is_fraud' not in df.columns:
            df['is_fraud'] = (np.random.random(
                size=len(df)) < 0.05).astype(int)

        # Create fraud prediction with slight errors from actual
        df['model_prediction'] = df['is_fraud'].copy()
        for i in range(len(df)):
            if df.loc[i, 'is_fraud'] == 1 and np.random.random() < 0.05:
                df.loc[i, 'model_prediction'] = 0
            elif df.loc[i, 'is_fraud'] == 0 and np.random.random() < 0.02:
                df.loc[i, 'model_prediction'] = 1

        # Add transaction_id if not present
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = [f"TX{i:06d}" for i in range(len(df))]

        # Add date column
        if 'date' not in df.columns:
            df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))

        # Ensure amount column exists
        if 'amount' not in df.columns and 'amt' in df.columns:
            df['amount'] = df['amt']

        # Add fraud probability if not present
        if 'fraud_probability' not in df.columns:
            df['fraud_probability'] = np.where(df['is_fraud'] == 1,
                                               np.random.beta(
                                                   2, 1, size=len(df)),
                                               np.random.beta(1, 5, size=len(df)))

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

    # Generate transaction amounts
    amounts = np.random.exponential(scale=100, size=n_samples)

    # Generate some features that correlate with fraud
    age = np.random.randint(18, 85, size=n_samples)
    credit_scores = np.random.normal(
        loc=650, scale=100, size=n_samples).astype(int)
    credit_scores = np.clip(credit_scores, 300, 850)

    # Generate fraud labels
    high_risk = (credit_scores < 600) | (amounts > 500)
    fraud_prob = np.where(high_risk,
                          np.random.beta(2, 5, size=n_samples),
                          np.random.beta(1, 20, size=n_samples))

    is_fraud = np.random.binomial(n=1, p=fraud_prob)

    # Create transaction types
    transaction_types = np.random.choice(
        ['online_purchase', 'in_store', 'atm_withdrawal', 'money_transfer'],
        size=n_samples
    )

    # Create countries
    countries = np.random.choice(
        ['USA', 'UK', 'Canada', 'Germany', 'France'],
        size=n_samples
    )

    # Create merchants
    merchants = [f"Merchant_{i}" for i in range(1, 21)]
    merchant_names = np.random.choice(merchants, size=n_samples)

    # Create model predictions with some errors
    model_predictions = np.copy(is_fraud)
    for i in range(n_samples):
        if is_fraud[i] == 1 and np.random.rand() < 0.05:
            model_predictions[i] = 0
        elif is_fraud[i] == 0 and np.random.rand() < 0.02:
            model_predictions[i] = 1

    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': [f"TX{i:06d}" for i in range(n_samples)],
        'date': dates,
        'amount': amounts,
        'age': age,
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

    return df


def create_metrics(df):
    """Create and display key metrics."""
    # Get the last 7 days of data
    last_7_days = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=7))]

    # Calculate metrics
    total_transactions = len(last_7_days)
    fraud_count = last_7_days['is_fraud'].sum()
    fraud_rate = fraud_count / total_transactions * 100
    avg_transaction = last_7_days['amount'].mean()

    # Detection metrics
    true_positive = ((last_7_days['model_prediction'] == 1) & (
        last_7_days['is_fraud'] == 1)).sum()
    false_negative = ((last_7_days['model_prediction'] == 0) & (
        last_7_days['is_fraud'] == 1)).sum()

    detection_rate = true_positive / \
        (true_positive + false_negative) * \
        100 if (true_positive + false_negative) > 0 else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transactions (Last 7 Days)", f"{total_transactions:,}")

    with col2:
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    with col3:
        st.metric("Avg. Transaction", f"${avg_transaction:.2f}")

    with col4:
        st.metric("Detection Rate", f"{detection_rate:.1f}%")


def plot_fraud_over_time(df):
    """Plot fraud trends over time."""
    # Group by date and calculate fraud metrics
    daily_fraud = df.groupby(df['date'].dt.date).agg(
        transactions=('is_fraud', 'count'),
        fraud_count=('is_fraud', 'sum')
    ).reset_index()

    daily_fraud['fraud_rate'] = daily_fraud['fraud_count'] / \
        daily_fraud['transactions'] * 100

    # Create figure
    fig = go.Figure()

    # Add transaction count
    fig.add_trace(go.Bar(
        x=daily_fraud['date'],
        y=daily_fraud['transactions'],
        name="Total Transactions",
        yaxis='y'
    ))

    # Add fraud rate
    fig.add_trace(go.Scatter(
        x=daily_fraud['date'],
        y=daily_fraud['fraud_rate'],
        name="Fraud Rate (%)",
        yaxis='y2',
        line=dict(color='red')
    ))

    fig.update_layout(
        title="Transaction Volume and Fraud Rate Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Transaction Count"),
        yaxis2=dict(title="Fraud Rate (%)", overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_confusion_matrix(df):
    """Plot confusion matrix for model performance."""
    # Calculate confusion matrix values
    true_positive = ((df['model_prediction'] == 1)
                     & (df['is_fraud'] == 1)).sum()
    false_positive = ((df['model_prediction'] == 1)
                      & (df['is_fraud'] == 0)).sum()
    false_negative = ((df['model_prediction'] == 0)
                      & (df['is_fraud'] == 1)).sum()
    true_negative = ((df['model_prediction'] == 0)
                     & (df['is_fraud'] == 0)).sum()

    # Create confusion matrix
    z = [[true_negative, false_positive], [false_negative, true_positive]]
    x = ['Predicted Genuine', 'Predicted Fraud']
    y = ['Actual Genuine', 'Actual Fraud']

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale='Blues',
        text=z,
        texttemplate="%{text}",
        textfont={"size": 16}
    ))

    fig.update_layout(
        title='Model Performance: Confusion Matrix',
        xaxis_title='Predicted Value',
        yaxis_title='Actual Value'
    )

    return fig


def plot_amount_distribution(df):
    """Plot the distribution of transaction amounts by fraud status."""
    fig = px.histogram(
        df,
        x="amount",
        color="is_fraud",
        nbins=50,
        opacity=0.7,
        barmode='overlay',
        range_x=[0, df['amount'].quantile(0.99)]
    )

    fig.update_layout(
        title="Distribution of Transaction Amounts by Fraud Status",
        xaxis_title="Transaction Amount ($)",
        yaxis_title="Count"
    )

    return fig


def main():
    # Set up page
    st.title("Fraud Detection Dashboard")

    # Display data and metrics
    df = load_demo_data()

    # Limit to 1000 samples for better performance
    if len(df) > 1000:
        df = df.sample(1000, random_state=42)

    # Create metrics
    create_metrics(df)

    # Main content display
    tabs = st.tabs(["Overview", "Fraud Over Time", "Model Performance"])

    with tabs[0]:  # Overview
        st.header("Fraud Overview")

        # Create two columns for the overview plots
        col1, col2 = st.columns(2)

        with col1:
            # Plot fraud over time
            fig = plot_fraud_over_time(df)
            st.plotly_chart(fig, use_container_width=True,
                            key="overview_fraud_time")

        with col2:
            # Plot amount distribution
            fig = plot_amount_distribution(df)
            st.plotly_chart(fig, use_container_width=True,
                            key="overview_amount_dist")

    with tabs[1]:  # Fraud Over Time
        st.header("Fraud Over Time Analysis")
        fig = plot_fraud_over_time(df)
        st.plotly_chart(fig, use_container_width=True,
                        key="fraud_time_analysis")

        # Daily transaction volume
        daily_totals = df.groupby(df['date'].dt.date)[
            'amount'].sum().reset_index()
        daily_count = df.groupby(
            df['date'].dt.date).size().reset_index(name='count')

        daily_data = pd.merge(daily_totals, daily_count, on='date')

        fig = px.line(daily_data, x='date', y='count',
                      title='Transaction Count by Day')
        st.plotly_chart(fig, use_container_width=True, key="daily_count")

        fig = px.line(daily_data, x='date', y='amount',
                      title='Transaction Amount by Day')
        st.plotly_chart(fig, use_container_width=True, key="daily_amount")

    with tabs[2]:  # Model Performance
        st.header("Model Performance Analysis")

        fig = plot_confusion_matrix(df)
        st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")

        # Fraud probability distribution
        if 'fraud_probability' in df.columns:
            fig = px.histogram(df, x='fraud_probability', color='is_fraud',
                               labels={'is_fraud': 'Actual Fraud'},
                               color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                               barmode='overlay',
                               opacity=0.7,
                               title='Distribution of Fraud Probability Scores')

            st.plotly_chart(fig, use_container_width=True,
                            key="fraud_prob_dist")


if __name__ == "__main__":
    main()
