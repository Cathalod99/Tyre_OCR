# test_system.py - Test script for the enhanced damage detection system
import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from advanced_model import AdvancedTireDamageModel, AdvancedTrainer, ModelEvaluator
        print("✅ advanced_model.py imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import advanced_model: {e}")
        return False
    
    try:
        from damage_inference import DamageDetector, DamageSeverityDetector
        print("✅ damage_inference.py imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import damage_inference: {e}")
        return False
    
    try:
        from integration import IntegratedTireAnalyzer, TireAnalysisResult
        print("✅ integration.py imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import integration: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🤖 Testing model creation...")
    
    try:
        from advanced_model import AdvancedTireDamageModel
        
        # Test ResNet50
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        print(f"✅ ResNet50 model created - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test EfficientNet-B0
        model = AdvancedTireDamageModel(num_classes=2, model_name='efficientnet_b0', pretrained=False)
        print(f"✅ EfficientNet-B0 model created - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test DenseNet121
        model = AdvancedTireDamageModel(num_classes=2, model_name='densenet121', pretrained=False)
        print(f"✅ DenseNet121 model created - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n📊 Testing data loading...")
    
    try:
        from enhanced_main import load_dataset
        
        # Check if data directory exists
        data_dir = Path("data")
        if not data_dir.exists():
            print("⚠️ Data directory not found - skipping data loading test")
            return True
        
        # Try to load dataset
        train_data, val_data, test_data = load_dataset(data_dir)
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Training: {len(train_data[0])} images")
        print(f"   Validation: {len(val_data[0])} images")
        print(f"   Test: {len(test_data[0])} images")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_transforms():
    """Test data augmentation transforms"""
    print("\n🔄 Testing data transforms...")
    
    try:
        from advanced_model import get_advanced_transforms
        
        train_transform, val_transform = get_advanced_transforms()
        print("✅ Transforms created successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test training transform
        train_tensor = train_transform(dummy_image)
        print(f"✅ Training transform: {train_tensor.shape}")
        
        # Test validation transform
        val_tensor = val_transform(dummy_image)
        print(f"✅ Validation transform: {val_tensor.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transform testing failed: {e}")
        return False

def test_inference():
    """Test inference functionality"""
    print("\n🔍 Testing inference...")
    
    try:
        from damage_inference import DamageDetector
        
        # Create a dummy model for testing
        from advanced_model import AdvancedTireDamageModel
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        
        # Save dummy model
        model_path = Path("test_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'model_name': 'resnet50', 'dropout_rate': 0.5}
        }, model_path)
        
        # Test detector initialization
        detector = DamageDetector(str(model_path), device='cpu')
        print("✅ DamageDetector initialized successfully")
        
        # Clean up
        model_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Inference testing failed: {e}")
        return False

def test_integration():
    """Test integration functionality"""
    print("\n🔗 Testing integration...")
    
    try:
        from integration import TireAnalysisResult
        
        # Test result creation
        result = TireAnalysisResult(
            image_path="test.jpg",
            tire_make="TestMake",
            tire_model="TestModel",
            is_damaged=False,
            damage_confidence=0.8
        )
        
        print("✅ TireAnalysisResult created successfully")
        print(f"   Make: {result.tire_make}")
        print(f"   Model: {result.tire_model}")
        print(f"   Damaged: {result.is_damaged}")
        print(f"   Confidence: {result.damage_confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration testing failed: {e}")
        return False

def test_device_availability():
    """Test device availability"""
    print("\n🖥️ Testing device availability...")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torch.__version__}")
    
    return True

def create_dummy_dataset():
    """Create a dummy dataset for testing"""
    print("\n📁 Creating dummy dataset for testing...")
    
    try:
        # Create directories
        data_dir = Path("test_data")
        (data_dir / "train" / "cracked").mkdir(parents=True, exist_ok=True)
        (data_dir / "train" / "normal").mkdir(parents=True, exist_ok=True)
        (data_dir / "val" / "cracked").mkdir(parents=True, exist_ok=True)
        (data_dir / "val" / "normal").mkdir(parents=True, exist_ok=True)
        (data_dir / "test" / "cracked").mkdir(parents=True, exist_ok=True)
        (data_dir / "test" / "normal").mkdir(parents=True, exist_ok=True)
        
        # Create dummy images
        for split in ["train", "val", "test"]:
            for category in ["cracked", "normal"]:
                for i in range(5):  # 5 images per category per split
                    # Create random image
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    img_path = data_dir / split / category / f"dummy_{i}.jpg"
                    cv2.imwrite(str(img_path), img)
        
        print(f"✅ Dummy dataset created in {data_dir}")
        print("   Structure:")
        print("   - train/cracked: 5 images")
        print("   - train/normal: 5 images")
        print("   - val/cracked: 5 images")
        print("   - val/normal: 5 images")
        print("   - test/cracked: 5 images")
        print("   - test/normal: 5 images")
        
        return True
        
    except Exception as e:
        print(f"❌ Dummy dataset creation failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    try:
        # Remove dummy dataset
        import shutil
        test_data_dir = Path("test_data")
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
            print("✅ Test data directory removed")
        
        # Remove test model
        test_model = Path("test_model.pth")
        if test_model.exists():
            test_model.unlink()
            print("✅ Test model file removed")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ENHANCED DAMAGE DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Data Transforms", test_transforms),
        ("Inference", test_inference),
        ("Integration", test_integration),
        ("Device Availability", test_device_availability),
        ("Dummy Dataset", create_dummy_dataset),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    print("\n🚀 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download dataset: python download_dataset.py")
    print("3. Train model: python train_advanced.py --model resnet50")
    print("4. Run inference: python damage_inference.py --model models/best_model.pth --image path/to/image.jpg")

if __name__ == "__main__":
    main()
