# enhanced_main.py - Enhanced tire damage detection with advanced features
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our advanced modules
from advanced_model import (
    AdvancedTireDamageModel, 
    DamageSeverityModel, 
    AdvancedTrainer, 
    ModelEvaluator,
    get_advanced_transforms
)

class TireDamageDataset(Dataset):
    """Enhanced dataset for tire damage classification with better error handling"""
    
    def __init__(self, image_paths, labels, transform=None, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Filter out invalid images
        self.valid_indices = []
        for i, path in enumerate(image_paths):
            if self._is_valid_image(path):
                self.valid_indices.append(i)
        
        print(f"📊 Dataset: {len(self.valid_indices)}/{len(image_paths)} valid images")
    
    def _is_valid_image(self, path):
        """Check if image is valid"""
        try:
            img = cv2.imread(str(path))
            return img is not None and img.shape[0] > 0 and img.shape[1] > 0
        except:
            return False
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        image_path = self.image_paths[actual_idx]
        
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            label = self.labels[actual_idx]
            return image, label
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            # Return a dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, 0

def load_dataset(data_dir, test_size=0.2, val_size=0.2):
    """Load and split the tire damage dataset"""
    print("🔄 Loading Tire Damage Dataset...")
    
    # Dataset paths
    cracked_dir = Path(data_dir) / "train" / "cracked"
    normal_dir = Path(data_dir) / "train" / "normal"
    
    image_paths = []
    labels = []
    
    # Load cracked tire images
    if cracked_dir.exists():
        for img_path in cracked_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(1)  # 1 for cracked
    
    # Load normal tire images
    if normal_dir.exists():
        for img_path in normal_dir.glob("*.jpg"):
            image_paths.append(img_path)
            labels.append(0)  # 0 for normal
    
    print(f"✅ Loaded {len(image_paths)} images")
    print(f"   - Cracked: {sum(labels)}")
    print(f"   - Normal: {len(labels) - sum(labels)}")
    
    # Split dataset
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=test_size + val_size, random_state=42, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size/(test_size + val_size), 
        random_state=42, stratify=temp_labels
    )
    
    print(f"📊 Dataset Split:")
    print(f"   - Training: {len(train_paths)} images")
    print(f"   - Validation: {len(val_paths)} images")
    print(f"   - Test: {len(test_paths)} images")
    
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)

def create_data_loaders(train_data, val_data, test_data, batch_size=32, num_workers=4):
    """Create data loaders with advanced transforms"""
    train_paths, train_labels = train_data
    val_paths, val_labels = val_data
    test_paths, test_labels = test_data
    
    # Get transforms
    train_transform, val_transform = get_advanced_transforms()
    
    # Create datasets
    train_dataset = TireDamageDataset(train_paths, train_labels, train_transform, augment=True)
    val_dataset = TireDamageDataset(val_paths, val_labels, val_transform)
    test_dataset = TireDamageDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def get_model_config(model_name='resnet50'):
    """Get model configuration"""
    configs = {
        'resnet50': {
            'model_name': 'resnet50',
            'optimizer': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'reduce_on_plateau',
            'grad_clip': 1.0,
            'early_stop_patience': 15,
            'use_class_weights': False,
            'dropout_rate': 0.5
        },
        'efficientnet_b0': {
            'model_name': 'efficientnet_b0',
            'optimizer': 'adamw',
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'grad_clip': 1.0,
            'early_stop_patience': 15,
            'use_class_weights': False,
            'dropout_rate': 0.3
        },
        'densenet121': {
            'model_name': 'densenet121',
            'optimizer': 'adam',
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler': 'reduce_on_plateau',
            'grad_clip': 1.0,
            'early_stop_patience': 15,
            'use_class_weights': False,
            'dropout_rate': 0.4
        }
    }
    
    return configs.get(model_name, configs['resnet50'])

def train_model(model_name='resnet50', epochs=50, batch_size=32, data_dir='data'):
    """Train the enhanced damage detection model"""
    print("🚀 Enhanced Tire Damage Detection System")
    print("=" * 50)
    
    # Set device - prioritize MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load dataset
    train_data, val_data, test_data = load_dataset(data_dir)
    
    if len(train_data[0]) == 0:
        print("❌ No training data found!")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size
    )
    
    # Get model configuration
    config = get_model_config(model_name)
    config['epochs'] = epochs
    
    # Initialize model
    model = AdvancedTireDamageModel(
        num_classes=2, 
        model_name=model_name, 
        pretrained=True,
        dropout_rate=config['dropout_rate']
    )
    model.to(device)
    
    print(f"🤖 Model: {model_name}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, device, config)
    
    # Train model
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, epochs)
    training_time = time.time() - start_time
    
    print(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    evaluator = ModelEvaluator(model, device)
    predictions, targets, probabilities = evaluator.evaluate(test_loader)
    
    # Generate comprehensive report
    results = evaluator.generate_report(predictions, targets, probabilities)
    
    # Save results
    save_results(model, history, results, config, model_name)
    
    # Plot results
    plot_results(history, results, model_name)
    
    return model, history, results

def save_results(model, history, results, config, model_name):
    """Save model and results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = Path(f"models/{model_name}_damage_model_{timestamp}.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'results': results
    }, model_path)
    
    # Save results as JSON
    results_path = Path(f"models/{model_name}_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': float(results['accuracy']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report'],
            'config': config,
            'timestamp': timestamp
        }, f, indent=2)
    
    print(f"💾 Model saved to {model_path}")
    print(f"📊 Results saved to {results_path}")

def plot_results(history, results, model_name):
    """Plot training results"""
    evaluator = ModelEvaluator(None, None)  # Dummy evaluator for plotting
    
    # Plot training history
    evaluator.plot_training_history(history, f"models/{model_name}_training_history.png")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        results['confusion_matrix'], 
        ['Normal', 'Cracked'],
        f"models/{model_name}_confusion_matrix.png"
    )

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Tire Damage Detection')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'densenet121'],
                       help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    
    args = parser.parse_args()
    
    # Set GPU if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"🖥️ Using GPU {args.gpu}")
    
    # Train model
    model, history, results = train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir
    )
    
    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()
