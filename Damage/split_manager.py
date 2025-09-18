# split_manager.py - Manage dataset splits with fixed seeds
import random
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(False)  # Keep False on MPS to avoid slowdowns

def create_fixed_splits(data_dir, test_size=0.2, val_size=0.2, seed=42, splits_file="splits.json"):
    """Create and save fixed dataset splits"""
    set_seeds(seed)
    
    # Dataset paths
    cracked_dir = Path(data_dir) / "train" / "cracked"
    normal_dir = Path(data_dir) / "train" / "normal"
    
    image_paths = []
    labels = []
    
    # Load cracked tire images
    if cracked_dir.exists():
        for img_path in cracked_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(1)  # 1 for cracked
    
    # Load normal tire images
    if normal_dir.exists():
        for img_path in normal_dir.glob("*.jpg"):
            image_paths.append(str(img_path))
            labels.append(0)  # 0 for normal
    
    print(f"📊 Total dataset: {len(image_paths)} images")
    print(f"   - Cracked: {sum(labels)}")
    print(f"   - Normal: {len(labels) - sum(labels)}")
    
    # Create splits with fixed seed
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=test_size + val_size, random_state=seed, stratify=labels
    )
    
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size/(test_size + val_size), 
        random_state=seed, stratify=temp_labels
    )
    
    # Save splits
    splits = {
        'train': {
            'paths': train_paths,
            'labels': train_labels
        },
        'val': {
            'paths': val_paths,
            'labels': val_labels
        },
        'test': {
            'paths': test_paths,
            'labels': test_labels
        },
        'metadata': {
            'total_images': len(image_paths),
            'cracked_count': sum(labels),
            'normal_count': len(labels) - sum(labels),
            'test_size': test_size,
            'val_size': val_size,
            'seed': seed
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"💾 Splits saved to {splits_file}")
    print(f"📊 Split sizes:")
    print(f"   - Training: {len(train_paths)} images")
    print(f"   - Validation: {len(val_paths)} images")
    print(f"   - Test: {len(test_paths)} images")
    
    return splits

def load_fixed_splits(splits_file="splits.json"):
    """Load fixed dataset splits"""
    if not Path(splits_file).exists():
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    print(f"📂 Loaded splits from {splits_file}")
    print(f"📊 Split sizes:")
    print(f"   - Training: {len(splits['train']['paths'])} images")
    print(f"   - Validation: {len(splits['val']['paths'])} images")
    print(f"   - Test: {len(splits['test']['paths'])} images")
    
    return splits

def get_class_weights(splits, device):
    """Calculate correct class weights"""
    train_labels = splits['train']['labels']
    class_counts = np.bincount(train_labels)
    
    # Calculate inverse frequency weights
    total_samples = len(train_labels)
    n_classes = len(class_counts)
    
    class_weights = []
    for count in class_counts:
        weight = total_samples / (n_classes * count)
        class_weights.append(weight)
    
    # Normalize weights
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum() * n_classes
    
    # Convert to tensor
    class_weights_tensor = torch.tensor(class_weights, device=device, dtype=torch.float32)
    
    print(f"📊 Class weights: {class_weights_tensor.cpu().numpy()}")
    print(f"   - Normal (0): {class_weights_tensor[0]:.4f}")
    print(f"   - Cracked (1): {class_weights_tensor[1]:.4f}")
    
    return class_weights_tensor
