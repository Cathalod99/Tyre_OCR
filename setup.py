#!/usr/bin/env python3
"""
Setup script for Tyre OCR system.
This script helps set up the environment and verify installation.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_environment_variables():
    """Check if required environment variables are set."""
    print("🔍 Checking environment variables...")
    
    required_vars = {
        'GOOGLE_APPLICATION_CREDENTIALS': 'Google Cloud Vision API credentials'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  {var}: {description}")
        else:
            print(f"✅ {var} is set")
    
    if missing_vars:
        print("⚠️ Missing environment variables:")
        for var in missing_vars:
            print(var)
        print("\nTo set them, run:")
        print("export GOOGLE_APPLICATION_CREDENTIALS='path/to/your/credentials.json'")
        return False
    
    return True

def check_model_files():
    """Check if required model files exist."""
    print("🔍 Checking model files...")
    
    model_path = Path("models/Tyre_Detect.onnx")
    if not model_path.exists():
        print(f"❌ YOLO model not found: {model_path}")
        print("Please ensure the Tyre_Detect.onnx file is in the models/ directory")
        return False
    
    print("✅ YOLO model found")
    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "sample_image", 
        "doc/img",
        "ML/__pycache__"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV imported")
    except ImportError:
        print("❌ OpenCV import failed")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported")
    except ImportError:
        print("❌ NumPy import failed")
        return False
    
    try:
        from google.cloud import vision
        print("✅ Google Cloud Vision imported")
    except ImportError:
        print("❌ Google Cloud Vision import failed")
        return False
    
    try:
        from ML.text_processor import TextProcessor
        print("✅ ML modules imported")
    except ImportError as e:
        print(f"❌ ML modules import failed: {e}")
        return False
    
    return True

def run_quick_test():
    """Run a quick test of the system."""
    print("🚀 Running quick test...")
    
    try:
        from ML.text_processor import TextProcessor
        processor = TextProcessor()
        
        # Test with sample text
        test_text = "MICHELIN PILOT SPORT 4 225/55R16 102V"
        result = processor.process_ocr_text(test_text)
        
        print("✅ Quick test passed")
        print(f"Sample result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Tyre OCR System Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Check environment variables
    if not check_environment_variables():
        success = False
    
    # Check model files
    if not check_model_files():
        success = False
    
    # Test imports
    if not test_imports():
        success = False
    
    # Run quick test
    if not run_quick_test():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place tire images in the 'sample_image' directory")
        print("2. Run: python main.py")
        print("3. Or test the system: python test_ml.py")
    else:
        print("❌ Setup completed with errors")
        print("Please fix the issues above and run setup again")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
