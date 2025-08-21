#!/usr/bin/env python3
"""
Test API Script
"""

import requests
import json
import time


def test_api():
    """Test the fraud detection API."""

    print("Testing Fraud Detection API")
    print("=" * 30)

    # Wait a moment for the API to start
    print("Waiting for API to start...")
    time.sleep(3)

    # Test health endpoint
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:5000")
        return

    # Test prediction endpoint
    print("\nTesting prediction endpoint...")

    # Sample transaction data
    sample_transaction = {
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
    }

    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json=sample_transaction,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(
                f"Transaction prediction: {'FRAUD' if result['predictions'] == 1 else 'GENUINE'}")
            print(f"Fraud probability: {result['probabilities']:.3f}")
            print(f"Model used: {result['model_used']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to prediction endpoint")
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")

    print("\nAPI test completed!")
    print("\nTo use the dashboard, open your browser and go to:")
    print("http://localhost:8501")


if __name__ == "__main__":
    test_api()
