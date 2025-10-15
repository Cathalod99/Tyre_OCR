#!/usr/bin/env python3
"""
Download and organize Kaggle Tire Quality Classification Dataset
"""

import kagglehub
import os
import shutil
from pathlib import Path
import json

def download_dataset():
    """Download the Kaggle tire quality classification dataset"""
    print("🚀 Downloading Kaggle Tire Quality Classification Dataset...")
    print("Dataset: warcoder/tyre-quality-classification")
    print("=" * 60)
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("warcoder/tyre-quality-classification")
        print(f"✅ Dataset downloaded to: {dataset_path}")
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def organize_dataset(dataset_path):
    """Organize the dataset into the expected structure"""
    print("\n🔄 Organizing dataset...")
    
    # Create target directories
    target_dir = Path("data")
    target_dir.mkdir(exist_ok=True)
    
    # Create train/val/test splits
    for split in ["train", "val", "test"]:
        for category in ["good", "bad"]:
            (target_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    # Find the actual data directory in the downloaded dataset
    dataset_dir = Path(dataset_path)
    
    # Look for common dataset structures
    possible_paths = [
        dataset_dir,
        dataset_dir / "dataset",
        dataset_dir / "data",
        dataset_dir / "tyre-quality-classification",
    ]
    
    data_source = None
    for path in possible_paths:
        if path.exists():
            # Look for image files
            jpg_files = list(path.rglob("*.jpg"))
            png_files = list(path.rglob("*.png"))
            if jpg_files or png_files:
                data_source = path
                break
    
    if data_source is None:
        print("❌ Could not find image files in downloaded dataset")
        print(f"Contents of {dataset_path}:")
        for item in dataset_dir.rglob("*"):
            if item.is_file():
                print(f"  - {item}")
        return False
    
    print(f"📁 Found data source: {data_source}")
    
    # Count files
    all_images = list(data_source.rglob("*.jpg")) + list(data_source.rglob("*.png"))
    print(f"📊 Total images found: {len(all_images)}")
    
    # Try to organize based on filename patterns
    good_images = []
    bad_images = []
    
    for img_path in all_images:
        filename = img_path.name.lower()
        if any(keyword in filename for keyword in ["good", "normal", "ok", "healthy"]):
            good_images.append(img_path)
        elif any(keyword in filename for keyword in ["bad", "crack", "damage", "defect", "faulty"]):
            bad_images.append(img_path)
        else:
            # If unclear, try to determine from parent directory
            parent_dir = img_path.parent.name.lower()
            if any(keyword in parent_dir for keyword in ["good", "normal", "ok", "healthy"]):
                good_images.append(img_path)
            elif any(keyword in parent_dir for keyword in ["bad", "crack", "damage", "defect", "faulty"]):
                bad_images.append(img_path)
            else:
                # Default to good if unclear
                good_images.append(img_path)
    
    print(f"📊 Organized images:")
    print(f"   - Good/Normal: {len(good_images)}")
    print(f"   - Bad/Damaged: {len(bad_images)}")
    
    # Copy files to organized structure
    # Use 70/15/15 split (train/val/test)
    import random
    random.seed(42)  # For reproducibility
    
    def split_and_copy(images, category):
        random.shuffle(images)
        total = len(images)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        print(f"   {category}: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
        
        # Copy files
        for i, img_path in enumerate(train_images):
            dest = target_dir / "train" / category / f"{category}_train_{i:04d}.jpg"
            shutil.copy2(img_path, dest)
        
        for i, img_path in enumerate(val_images):
            dest = target_dir / "val" / category / f"{category}_val_{i:04d}.jpg"
            shutil.copy2(img_path, dest)
        
        for i, img_path in enumerate(test_images):
            dest = target_dir / "test" / category / f"{category}_test_{i:04d}.jpg"
            shutil.copy2(img_path, dest)
    
    # Organize good images
    if good_images:
        split_and_copy(good_images, "good")
    
    # Organize bad images  
    if bad_images:
        split_and_copy(bad_images, "bad")
    
    # Create metadata
    metadata = {
        "dataset_name": "Kaggle Tire Quality Classification",
        "source": "warcoder/tyre-quality-classification",
        "total_images": len(all_images),
        "classes": {
            "good": len(good_images),
            "bad": len(bad_images)
        },
        "splits": {
            "train": {
                "good": len(good_images[:int(len(good_images) * 0.7)]) if good_images else 0,
                "bad": len(bad_images[:int(len(bad_images) * 0.7)]) if bad_images else 0
            },
            "val": {
                "good": len(good_images[int(len(good_images) * 0.7):int(len(good_images) * 0.85)]) if good_images else 0,
                "bad": len(bad_images[int(len(bad_images) * 0.7):int(len(bad_images) * 0.85)]) if bad_images else 0
            },
            "test": {
                "good": len(good_images[int(len(good_images) * 0.85):]) if good_images else 0,
                "bad": len(bad_images[int(len(bad_images) * 0.85):]) if bad_images else 0
            }
        },
        "split_ratio": "70/15/15",
        "random_seed": 42,
        "synthetic": False
    }
    
    # Save metadata
    with open(target_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"📁 Dataset location: {target_dir.absolute()}")
    print(f"📄 Metadata saved: {target_dir / 'dataset_metadata.json'}")
    
    return True

def main():
    """Main function"""
    print("📥 Kaggle Tire Quality Classification Dataset Downloader")
    print("=" * 60)
    
    # Download dataset
    dataset_path = download_dataset()
    if not dataset_path:
        return
    
    # Organize dataset
    success = organize_dataset(dataset_path)
    if success:
        print("\n🎉 Dataset download and organization complete!")
        print("\nNext steps:")
        print("1. Verify the dataset structure in Damage/data/")
        print("2. Run your training scripts")
        print("3. The dataset is now ready for use!")
    else:
        print("\n❌ Dataset organization failed. Please check the downloaded files manually.")

if __name__ == "__main__":
    main()