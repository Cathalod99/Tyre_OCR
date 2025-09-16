#!/usr/bin/env python3
"""
Test script for the ML-based tire OCR system.
This script tests the individual ML modules and the integrated system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ML.text_processor import TextProcessor
from ML.tire_classifier import TireClassifier
from ML.size_extractor import SizeExtractor
from ML.dot_extractor import DOTExtractor

def test_individual_models():
    """Test individual ML models."""
    print("🧪 Testing individual ML models...")
    
    # Test tire classifier
    print("\n1. Testing Tire Classifier:")
    classifier = TireClassifier()
    test_texts = [
        "MICHELIN PILOT SPORT 4 225/55R16",
        "BRIDGESTONE TURANZA T005 205/60R16",
        "HANKOOK VENTUS PRIME4 225/65R17"
    ]
    
    for text in test_texts:
        result = classifier.predict(text)
        print(f"   Input: {text}")
        print(f"   Output: {result}")
        print()
    
    # Test size extractor
    print("2. Testing Size Extractor:")
    size_extractor = SizeExtractor()
    for text in test_texts:
        result = size_extractor.predict_size_info(text)
        print(f"   Input: {text}")
        print(f"   Output: {result}")
        print()
    
    # Test DOT extractor
    print("3. Testing DOT Extractor:")
    dot_extractor = DOTExtractor()
    dot_test_texts = [
        "DOT 6Y87 KY7L 4220",
        "DOT 15MH1 18HB 4222",
        "DOT 4T00 00HB 2024"
    ]
    
    for text in dot_test_texts:
        result = dot_extractor.predict_dot_info(text)
        print(f"   Input: {text}")
        print(f"   Output: {result}")
        print()

def test_integrated_system():
    """Test the integrated TextProcessor system."""
    print("🔧 Testing integrated TextProcessor system...")
    
    try:
        processor = TextProcessor()
        print("✓ TextProcessor initialized successfully")
        
        # Test with sample OCR text
        sample_ocr = """
        HANKOOK
        VENTUS PRIME4
        225/65R17 102H
        DOT 15MH1 18HB 4222
        MADE IN KOREA
        """
        
        result = processor.process_ocr_text(sample_ocr)
        print("\n📋 Sample OCR Input:")
        print(sample_ocr)
        print("\n📊 Extracted Information:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        # Validate result
        is_valid = processor.validate_result(result)
        print(f"\n✅ Result validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Get model info
        model_info = processor.get_model_info()
        print(f"\n📈 Model Status:")
        for model, status in model_info.items():
            print(f"   {model}: {status}")
            
    except Exception as e:
        print(f"❌ Error testing integrated system: {e}")
        return False
    
    return True

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n🛡️ Testing error handling...")
    
    processor = TextProcessor()
    
    # Test empty input
    result = processor.process_ocr_text("")
    print(f"Empty input result: {result}")
    
    # Test None input
    result = processor.process_ocr_text(None)
    print(f"None input result: {result}")
    
    # Test invalid text
    result = processor.process_ocr_text("This is not tire information")
    print(f"Invalid text result: {result}")

def main():
    """Run all tests."""
    print("🚀 Starting ML-based Tire OCR System Tests")
    print("=" * 50)
    
    try:
        # Test individual models
        test_individual_models()
        
        # Test integrated system
        success = test_integrated_system()
        
        # Test error handling
        test_error_handling()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("\n💡 To run the main system:")
            print("   python main.py")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
