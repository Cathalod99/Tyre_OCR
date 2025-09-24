#!/usr/bin/env python3
"""
Minimal setup script for Tyre OCR ML System
This script sets up the environment and credentials needed to run main_ml.py
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up the minimal environment for running main_ml.py"""
    
    print("🚀 Setting up minimal Tyre OCR ML environment...")
    
    # Check if we're in the right directory
    if not Path("main_ml.py").exists():
        print("❌ Error: main_ml.py not found. Please run this script from the project root.")
        return False
    
    # Set up Google Cloud credentials
    credentials_file = "google_credentials.json"
    if Path(credentials_file).exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(Path(credentials_file).absolute())
        print(f"✅ Google Cloud credentials set: {credentials_file}")
    else:
        print("⚠️ Warning: google_credentials.json not found. OCR will not work without proper credentials.")
    
    # Create necessary directories
    Path("doc/img").mkdir(parents=True, exist_ok=True)
    print("✅ Created doc/img directory")
    
    # Check if sample images exist
    sample_dir = Path("sample_image")
    if sample_dir.exists() and list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg")):
        print("✅ Sample images found")
    else:
        print("⚠️ Warning: No sample images found in sample_image/ directory")
    
    # Check if models exist
    model_files = ["models/Tyre_Detect.onnx", "models/tire_classifier.pkl"]
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Model found: {model_file}")
        else:
            print(f"❌ Missing model: {model_file}")
    
    print("\n🎉 Setup complete!")
    print("\nTo run the system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run: python main_ml.py")
    print("\nMake sure to place tire images in the sample_image/ directory")
    
    return True

if __name__ == "__main__":
    setup_environment()
