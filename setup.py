"""
SecurePay Fraud Detection - Setup Script
----------------------------------------
This script helps set up the SecurePay Fraud Detection system.
"""

import os
import sys
import subprocess
import platform

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)

def run_command(command):
    """Run a shell command and print the output."""
    print(f"> {command}")
    try:
        process = subprocess.run(command, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_python_version():
    """Check if the Python version is compatible."""
    print_header("Checking Python version")
    
    python_version = sys.version_info
    print(f"Detected Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor < 8:
        print("ERROR: Python 3.8+ is required. Please upgrade your Python installation.")
        sys.exit(1)
    
    if python_version.major == 3 and python_version.minor >= 13:
        print("Warning: Python 3.13+ detected. Some packages may have compatibility issues.")
        print("Recommended: Use Python 3.8-3.11 for best compatibility.")
        answer = input("Do you want to continue anyway? (y/n): ").lower()
        if answer != 'y':
            print("Setup aborted. Please install a compatible Python version.")
            sys.exit(1)

def create_directories():
    """Create necessary directories if they don't exist."""
    print_header("Creating directory structure")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/feedback",
        "models",
        "models/figures",
        "notebooks",
        "tests/unit",
        "tests/integration",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating {directory}/")
            os.makedirs(directory, exist_ok=True)
        else:
            print(f"{directory}/ already exists")

def install_dependencies():
    """Install Python dependencies with error handling."""
    print_header("Installing dependencies")
    
    # First try to upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Try to install basic numerical packages first
    run_command(f"{sys.executable} -m pip install numpy pandas --upgrade")
    
    # Then install the rest of the requirements
    success = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    if not success:
        print("\nSome packages failed to install. Trying alternative approach...")
        # Try installing one by one
        with open('requirements.txt', 'r') as f:
            packages = [line.strip() for line in f if line.strip()]
        
        for package in packages:
            run_command(f"{sys.executable} -m pip install {package}")
    
    print("\nDependency installation completed. Some warnings may be normal.")

def install_dev_dependencies():
    """Install development dependencies for testing and linting."""
    print_header("Installing development dependencies")
    
    dev_packages = ["pytest", "pytest-cov", "black", "flake8"]
    dev_deps_str = " ".join(dev_packages)
    run_command(f"{sys.executable} -m pip install {dev_deps_str}")

def copy_data_files():
    """Copy the data files to the correct location."""
    print_header("Setting up data files")
    
    # Check for creditcard.csv file
    if os.path.exists("creditcard.csv"):
        print("Copying creditcard.csv to data/raw/")
        import shutil
        shutil.copy("creditcard.csv", "data/raw/creditcard.csv")
        return True
        
    # Try to download a sample dataset if file not found
    print("creditcard.csv not found. Attempting to download a sample dataset...")
    try:
        # Create data/raw directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        
        # Try to download from various sources
        urls = [
            "https://raw.githubusercontent.com/andysingal/DataSet/master/creditcard_sample.csv",
            "https://raw.githubusercontent.com/mlabonne/LeadingEdgeFraudDetection/main/creditcard_sample.csv"
        ]
        
        for url in urls:
            try:
                if platform.system() == "Windows":
                    download_command = f"curl -o data/raw/creditcard_sample.csv {url}"
                else:
                    download_command = f"wget -O data/raw/creditcard_sample.csv {url}"
                
                if run_command(download_command):
                    print(f"Sample dataset downloaded to data/raw/creditcard_sample.csv from {url}")
                    return True
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
        
        print("Warning: Could not download sample dataset. You'll need to place your dataset in the data/raw/ directory manually.")
        return False
    except Exception as e:
        print(f"Error setting up data files: {e}")
        print("You'll need to place your dataset in the data/raw/ directory manually.")
        return False

def create_ci_cd_files():
    """Create CI/CD configuration files."""
    print_header("Setting up CI/CD infrastructure")
    
    # GitHub Actions workflow
    github_workflow = """\
name: SecurePay CI/CD

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    - name: Format check with black
      run: |
        black --check src
    - name: Test with pytest
      run: |
        pytest --cov=src tests/unit/

  build-and-publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && (github.ref == 'refs/heads/main' || github.ref == 'refs/heads/master')
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build wheel setuptools
    - name: Build package
      run: |
        python -m build
    - name: Store artifacts
      uses: actions/upload-artifact@v3
      with:
        name: secureplay-fraud-detection
        path: dist/
"""
    
    # Create GitHub workflow directory
    os.makedirs('.github/workflows', exist_ok=True)
    
    with open('.github/workflows/ci-cd.yml', 'w') as f:
        f.write(github_workflow)
    
    # Create Docker configuration
    dockerfile = """\
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/feedback models logs

# Set environment variables
ENV PYTHONPATH=/app

# Run the API by default
CMD ["python", "src/api/app.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    # Create docker-compose.yml
    docker_compose = """\
version: '3'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    command: python src/api/app.py
    environment:
      - PYTHONUNBUFFERED=1

  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    command: streamlit run src/dashboard/main.py
    environment:
      - PYTHONUNBUFFERED=1
    depends_on:
      - api
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    
    print("CI/CD files created successfully")

def print_next_steps():
    """Print instructions for next steps."""
    print_header("Setup Complete!")
    print("Next steps:")
    print("1. Process the data:   python src/features/feature_engineering.py")
    print("2. Train the models:   python src/models/train_models.py")
    print("3. Start the API:      python src/api/app.py")
    print("4. Run the dashboard:  streamlit run src/dashboard/main.py")
    print("5. Run tests:          pytest")
    print("\nDocker commands:")
    print("- Build and start all services:  docker-compose up --build")
    print("- Run API service only:          docker-compose up api")
    print("- Run dashboard only:            docker-compose up dashboard")
    print("\nFor more information, refer to the README.md file.")

def main():
    """Main function to run the setup process."""
    print_header("SecurePay Fraud Detection - Setup")
    
    # Check Python version
    check_python_version()
    
    # Create necessary directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Ask if user wants to install development dependencies
    install_dev = input("\nDo you want to install development dependencies? (y/n): ").lower()
    if install_dev.startswith('y'):
        install_dev_dependencies()
    # Copy data files
    copy_data_files()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 