# demo_enhanced_system.py - Demonstration of the enhanced damage detection system
import sys
import os
from pathlib import Path
import torch
import numpy as np
import cv2
import time
import json
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def demo_model_training():
    """Demonstrate model training capabilities"""
    print("🚀 DEMO: Model Training")
    print("=" * 40)
    
    try:
        from advanced_model import AdvancedTireDamageModel, AdvancedTrainer, get_advanced_transforms
        from enhanced_main import load_dataset, create_data_loaders
        
        # Check if data exists
        data_dir = Path("data")
        if not data_dir.exists():
            print("⚠️ Data directory not found. Please run download_dataset.py first.")
            return False
        
        print("📊 Loading dataset...")
        train_data, val_data, test_data = load_dataset(data_dir)
        
        if len(train_data[0]) == 0:
            print("⚠️ No training data found. Please organize your dataset.")
            return False
        
        print(f"✅ Dataset loaded: {len(train_data[0])} training images")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, batch_size=16
        )
        
        # Initialize model
        print("🤖 Initializing ResNet50 model...")
        model = AdvancedTireDamageModel(
            num_classes=2,
            model_name='resnet50',
            pretrained=True,
            dropout_rate=0.5
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🖥️ Using device: {device}")
        
        # Training configuration
        config = {
            'model_name': 'resnet50',
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'reduce_on_plateau',
            'grad_clip': 1.0,
            'early_stop_patience': 5,  # Short for demo
            'use_class_weights': False,
            'dropout_rate': 0.5
        }
        
        # Initialize trainer
        trainer = AdvancedTrainer(model, device, config)
        
        print("🚀 Starting training (5 epochs for demo)...")
        start_time = time.time()
        history = trainer.train(train_loader, val_loader, epochs=5)
        training_time = time.time() - start_time
        
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        print("📊 Evaluating model...")
        from advanced_model import ModelEvaluator
        evaluator = ModelEvaluator(model, device)
        predictions, targets, probabilities = evaluator.evaluate(test_loader)
        results = evaluator.generate_report(predictions, targets, probabilities)
        
        print(f"✅ Demo training completed successfully!")
        print(f"   Test Accuracy: {results['accuracy']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        return False

def demo_inference():
    """Demonstrate inference capabilities"""
    print("\n🔍 DEMO: Damage Detection Inference")
    print("=" * 40)
    
    try:
        from damage_inference import DamageDetector
        
        # Create a dummy model for demo
        from advanced_model import AdvancedTireDamageModel
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        
        # Save dummy model
        model_path = Path("demo_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'model_name': 'resnet50', 'dropout_rate': 0.5}
        }, model_path)
        
        # Initialize detector
        detector = DamageDetector(str(model_path), device='cpu', confidence_threshold=0.5)
        print("✅ Damage detector initialized")
        
        # Create dummy test images
        test_images = []
        for i in range(3):
            # Create random test image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img_path = f"demo_test_{i}.jpg"
            cv2.imwrite(img_path, img)
            test_images.append(img_path)
        
        print(f"📸 Created {len(test_images)} test images")
        
        # Test single image prediction
        print("\n🔍 Single image prediction:")
        result = detector.predict_single(test_images[0])
        print(f"   Image: {result['image_path']}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Processing time: {result['processing_time']:.3f}s")
        
        # Test batch prediction
        print("\n📊 Batch prediction:")
        results = detector.predict_batch(test_images)
        
        for i, result in enumerate(results):
            print(f"   Image {i+1}: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        
        # Test statistics
        stats = detector.get_statistics(results)
        print(f"\n📈 Batch Statistics:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Average confidence: {stats['average_confidence']:.3f}")
        print(f"   Average processing time: {stats['average_processing_time']:.3f}s")
        
        # Cleanup
        for img_path in test_images:
            Path(img_path).unlink()
        model_path.unlink()
        
        print("✅ Inference demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference demo failed: {e}")
        return False

def demo_integration():
    """Demonstrate integration with OCR pipeline"""
    print("\n🔗 DEMO: OCR Integration")
    print("=" * 40)
    
    try:
        from integration import IntegratedTireAnalyzer, TireAnalysisResult
        
        # Create dummy model for demo
        from advanced_model import AdvancedTireDamageModel
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        
        model_path = Path("demo_integration_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'model_name': 'resnet50', 'dropout_rate': 0.5}
        }, model_path)
        
        # Initialize analyzer (OCR will be mocked)
        analyzer = IntegratedTireAnalyzer(str(model_path), damage_confidence_threshold=0.5)
        print("✅ Integrated analyzer initialized")
        
        # Create dummy test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_img_path = "demo_tire.jpg"
        cv2.imwrite(test_img_path, test_img)
        
        print(f"📸 Created test tire image: {test_img_path}")
        
        # Analyze tire
        print("\n🔍 Analyzing tire...")
        result = analyzer.analyze_tire(test_img_path)
        
        print(f"📊 Analysis Results:")
        print(f"   Image: {Path(result.image_path).name}")
        print(f"   OCR Info: {result.tire_make} {result.tire_model} {result.tire_size}")
        print(f"   DOT Code: {result.dot_code}")
        print(f"   Damage: {'DAMAGED' if result.is_damaged else 'NORMAL'}")
        print(f"   Confidence: {result.damage_confidence:.3f}")
        print(f"   Risk Level: {result.risk_level}")
        print(f"   Recommendation: {result.safety_recommendation}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        
        if result.errors:
            print(f"   Errors: {'; '.join(result.errors)}")
        
        # Test batch analysis
        print("\n📊 Batch analysis:")
        test_images = [test_img_path]
        results = analyzer.analyze_batch(test_images)
        
        # Generate report
        report = analyzer.generate_comprehensive_report(results)
        print("\n📄 Generated comprehensive report:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        # Cleanup
        Path(test_img_path).unlink()
        model_path.unlink()
        
        print("✅ Integration demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration demo failed: {e}")
        return False

def demo_advanced_features():
    """Demonstrate advanced features"""
    print("\n⚡ DEMO: Advanced Features")
    print("=" * 40)
    
    try:
        from advanced_model import AdvancedTireDamageModel, get_advanced_transforms
        
        # Test different model architectures
        print("🤖 Testing different model architectures:")
        
        models = ['resnet50', 'efficientnet_b0', 'densenet121']
        for model_name in models:
            model = AdvancedTireDamageModel(num_classes=2, model_name=model_name, pretrained=False)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"   {model_name}: {param_count:,} parameters")
        
        # Test data augmentation
        print("\n🔄 Testing data augmentation:")
        train_transform, val_transform = get_advanced_transforms()
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test transforms
        train_tensor = train_transform(dummy_img)
        val_tensor = val_transform(dummy_img)
        
        print(f"   Training transform: {train_tensor.shape}")
        print(f"   Validation transform: {val_tensor.shape}")
        
        # Test model freezing/unfreezing
        print("\n🔒 Testing model freezing:")
        model = AdvancedTireDamageModel(num_classes=2, model_name='resnet50', pretrained=False)
        
        # Count trainable parameters before freezing
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters before freezing: {trainable_before:,}")
        
        # Freeze backbone
        model.freeze_backbone()
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters after freezing: {trainable_after:,}")
        
        # Unfreeze backbone
        model.unfreeze_backbone()
        trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Trainable parameters after unfreezing: {trainable_unfrozen:,}")
        
        print("✅ Advanced features demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Advanced features demo failed: {e}")
        return False

def main():
    """Run all demonstrations"""
    print("🎬 ENHANCED DAMAGE DETECTION SYSTEM - DEMONSTRATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    demos = [
        ("Model Training", demo_model_training),
        ("Inference", demo_inference),
        ("OCR Integration", demo_integration),
        ("Advanced Features", demo_advanced_features),
    ]
    
    passed = 0
    total = len(demos)
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            if demo_func():
                passed += 1
                print(f"✅ {demo_name} demo completed successfully")
            else:
                print(f"⚠️ {demo_name} demo completed with warnings")
        except Exception as e:
            print(f"❌ {demo_name} demo failed: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 DEMO SUMMARY: {passed}/{total} demos completed successfully")
    
    if passed == total:
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("\n🚀 The enhanced damage detection system is ready to use!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download dataset: python download_dataset.py")
        print("3. Train model: python train_advanced.py --model resnet50 --epochs 50")
        print("4. Run inference: python damage_inference.py --model models/best_model.pth --image path/to/image.jpg")
        print("5. Integrated analysis: python integration.py --damage_model models/best_model.pth --image path/to/tire.jpg")
    else:
        print("⚠️ Some demos had issues. Please check the errors above.")
    
    print(f"\n📚 For detailed documentation, see: README_ENHANCED.md")
    print(f"🧪 For system tests, run: python test_system.py")

if __name__ == "__main__":
    main()
