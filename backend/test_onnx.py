#!/usr/bin/env python3
"""
Test script to verify ONNX Runtime installation
"""
import sys
import os

def test_onnx_runtime():
    """Test ONNX Runtime installation and functionality"""
    try:
        print("Testing ONNX Runtime installation...")
        print(f"Python version: {sys.version}")
        print(f"Python path: {sys.path}")
        
        # Test import
        import onnxruntime
        print(f"✅ ONNX Runtime version: {onnxruntime.__version__}")
        print(f"✅ Available providers: {onnxruntime.get_available_providers()}")
        print(f"✅ Installation path: {onnxruntime.__file__}")
        
        # Test model loading
        model_path = "/app/models/Tyre_Detect.onnx"
        if os.path.exists(model_path):
            session = onnxruntime.InferenceSession(model_path)
            print(f"✅ YOLO model loaded successfully")
            print(f"✅ Model input names: {session.get_inputs()[0].name}")
            print(f"✅ Model output names: {[output.name for output in session.get_outputs()]}")
        else:
            print(f"❌ YOLO model not found at: {model_path}")
            
        return True
        
    except ImportError as e:
        print(f"❌ ONNX Runtime import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ONNX Runtime test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_onnx_runtime()
    sys.exit(0 if success else 1)
