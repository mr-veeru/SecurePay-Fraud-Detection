#!/usr/bin/env python3
"""
Fraud Detection System - Main Runner
"""

import os
import sys
import subprocess
import time


def print_header():
    """Print the header."""
    print("=" * 50)
    print("🛡️  FRAUD DETECTION SYSTEM")
    print("=" * 50)
    print()


def check_data():
    """Check if data exists."""
    if not os.path.exists('data/creditcard.csv'):
        print("❌ Error: data/creditcard.csv not found!")
        print("\n📋 To get data, you have several options:")
        print("   1. Generate sample data: python generate_sample_data.py")
        print("   2. Download from Kaggle: kaggle datasets download -d mlg-ulb/creditcardfraud")
        print("   3. Place your own CSV file in data/creditcard.csv")
        print("\n💡 For testing, we recommend option 1 (generate sample data)")
        return False
    return True


def run_feature_engineering():
    """Run feature engineering."""
    print("🔧 Running Feature Engineering...")
    try:
        result = subprocess.run([sys.executable, 'feature_engineering.py'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Feature engineering completed successfully!")
            return True
        else:
            print(f"❌ Feature engineering failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running feature engineering: {e}")
        return False


def run_model_training():
    """Run model training."""
    print("🤖 Training Models...")
    try:
        result = subprocess.run([sys.executable, 'train_models.py'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running model training: {e}")
        return False


def test_api():
    """Test the API."""
    print("🧪 Testing API...")
    try:
        result = subprocess.run([sys.executable, 'test_api.py'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ API test completed successfully!")
            return True
        else:
            print(f"❌ API test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False


def start_api_server():
    """Start the API server."""
    print("🚀 Starting API Server...")
    try:
        # Start API server in background
        api_process = subprocess.Popen([sys.executable, 'api_server.py'])
        time.sleep(3)  # Wait for server to start
        print("✅ API server started on http://localhost:5000")
        return api_process
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None


def start_dashboard():
    """Start the dashboard."""
    print("📊 Starting Dashboard...")
    try:
        # Start dashboard in background
        dashboard_process = subprocess.Popen(
            [sys.executable, '-m', 'streamlit', 'run', 'dashboard.py'])
        time.sleep(5)  # Wait for dashboard to start
        print("✅ Dashboard started on http://localhost:8501")
        return dashboard_process
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None


def main():
    """Main function."""
    print_header()

    # Check if data exists
    if not check_data():
        return

    # Check if processed data exists
    if not os.path.exists('data/processed/selected_features.csv'):
        print("📊 Processed data not found. Running feature engineering...")
        if not run_feature_engineering():
            return

    # Check if models exist
    if not os.path.exists('models/random_forest.pkl'):
        print("🤖 Models not found. Running model training...")
        if not run_model_training():
            return

    print("🎉 All components ready!")
    print()

    # Menu
    while True:
        print("Choose an option:")
        print("1. Start API Server")
        print("2. Start Dashboard")
        print("3. Test API")
        print("4. Run Feature Engineering")
        print("5. Run Model Training")
        print("6. Start Both (API + Dashboard)")
        print("7. Exit")
        print()

        choice = input("Enter your choice (1-7): ").strip()

        if choice == '1':
            api_process = start_api_server()
            if api_process:
                input("Press Enter to stop the API server...")
                api_process.terminate()
                print("API server stopped.")

        elif choice == '2':
            dashboard_process = start_dashboard()
            if dashboard_process:
                input("Press Enter to stop the dashboard...")
                dashboard_process.terminate()
                print("Dashboard stopped.")

        elif choice == '3':
            test_api()

        elif choice == '4':
            run_feature_engineering()

        elif choice == '5':
            run_model_training()

        elif choice == '6':
            print("🚀 Starting both API and Dashboard...")
            api_process = start_api_server()
            dashboard_process = start_dashboard()

            if api_process and dashboard_process:
                print()
                print("🎉 Both services are running!")
                print("📊 Dashboard: http://localhost:8501")
                print("🔌 API: http://localhost:5000")
                print()
                input("Press Enter to stop both services...")
                api_process.terminate()
                dashboard_process.terminate()
                print("Both services stopped.")

        elif choice == '7':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice. Please enter a number between 1 and 7.")

        print()


if __name__ == "__main__":
    main()
