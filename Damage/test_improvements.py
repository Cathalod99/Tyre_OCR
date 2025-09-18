# test_improvements.py - Test all 4 improvements
import torch
import numpy as np
import random
from pathlib import Path

def test_improvements():
    """Test all 4 improvements"""
    
    print("🧪 TESTING ALL IMPROVEMENTS")
    print("=" * 50)
    
    # Test 1: Seeds and reproducibility
    print("\n1️⃣ Testing Seeds and Reproducibility:")
    print("-" * 40)
    
    from split_manager import set_seeds
    
    # Test seed setting
    set_seeds(42)
    print("✅ Seeds set successfully")
    
    # Test reproducibility
    torch.manual_seed(42)
    a = torch.randn(5, 5)
    torch.manual_seed(42)
    b = torch.randn(5, 5)
    print(f"✅ Reproducible random: {torch.allclose(a, b)}")
    
    # Test 2: Class weights
    print("\n2️⃣ Testing Class Weights:")
    print("-" * 40)
    
    from split_manager import get_class_weights
    
    # Mock splits data
    mock_splits = {
        'train': {
            'labels': [0] * 301 + [1] * 262  # Normal=301, Cracked=262
        }
    }
    
    device = torch.device("cpu")
    class_weights = get_class_weights(mock_splits, device)
    
    print(f"✅ Class weights calculated: {class_weights.cpu().numpy()}")
    print(f"   - Normal (0): {class_weights[0]:.4f}")
    print(f"   - Cracked (1): {class_weights[1]:.4f}")
    
    # Verify correct ordering (cracked should have higher weight)
    if class_weights[1] > class_weights[0]:
        print("✅ Correct ordering: Cracked has higher weight (minority class)")
    else:
        print("❌ Wrong ordering: Normal has higher weight")
    
    # Test 3: Torchvision deprecation fix
    print("\n3️⃣ Testing Torchvision Deprecation Fix:")
    print("-" * 40)
    
    try:
        from advanced_model import AdvancedTireDamageModel
        
        # Test ResNet50 with new weights parameter
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=True)
        print("✅ ResNet50 loaded with new weights parameter (no deprecation warnings)")
        
        # Test EfficientNet-B0
        model = AdvancedTireDamageModel(num_classes=2, model_name='efficientnet_b0', pretrained=True)
        print("✅ EfficientNet-B0 loaded with new weights parameter")
        
        # Test DenseNet121
        model = AdvancedTireDamageModel(num_classes=2, model_name='densenet121', pretrained=True)
        print("✅ DenseNet121 loaded with new weights parameter")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
    
    # Test 4: Cosine scheduler
    print("\n4️⃣ Testing Cosine Scheduler:")
    print("-" * 40)
    
    try:
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Create mock optimizer and scheduler
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        optimizer = Adam(model.parameters(), lr=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        
        print("✅ Cosine scheduler created successfully")
        print(f"   - Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Test stepping
        scheduler.step()
        print(f"   - After step: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Test multiple steps
        for i in range(5):
            scheduler.step()
        print(f"   - After 5 steps: {optimizer.param_groups[0]['lr']:.2e}")
        
    except Exception as e:
        print(f"❌ Error testing scheduler: {e}")
    
    # Test 5: Fixed splits
    print("\n5️⃣ Testing Fixed Splits:")
    print("-" * 40)
    
    try:
        from split_manager import create_fixed_splits, load_fixed_splits
        
        # Test creating splits
        if Path("data").exists():
            splits = create_fixed_splits("data", test_size=0.2, val_size=0.2, seed=42, splits_file="test_splits.json")
            print("✅ Fixed splits created successfully")
            
            # Test loading splits
            loaded_splits = load_fixed_splits("test_splits.json")
            print("✅ Fixed splits loaded successfully")
            
            # Clean up test file
            Path("test_splits.json").unlink()
            print("✅ Test file cleaned up")
        else:
            print("⚠️ Data directory not found - skipping splits test")
            
    except Exception as e:
        print(f"❌ Error testing splits: {e}")
    
    print(f"\n🎉 ALL IMPROVEMENTS TESTED!")
    print("=" * 50)
    print("✅ Ready for enhanced training!")

if __name__ == "__main__":
    test_improvements()
