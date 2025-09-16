# download_dataset.py - Download Harvard Tire Damage Dataset
import os
import requests
import zipfile
from pathlib import Path
import time

def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"🔄 Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
    
    print(f"\n✅ Download complete: {filename}")

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"🔄 Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"✅ Extraction complete to {extract_to}")

def organize_dataset(data_dir):
    """Organize the dataset into cracked/normal folders"""
    print("🔄 Organizing dataset...")
    
    data_path = Path(data_dir)
    
    # Create organized directories
    cracked_dir = data_path / "cracked"
    normal_dir = data_path / "normal"
    
    cracked_dir.mkdir(exist_ok=True)
    normal_dir.mkdir(exist_ok=True)
    
    # Note: The actual organization will depend on the dataset structure
    # This is a placeholder for the organization logic
    print("ℹ️  Please manually organize the dataset into:")
    print(f"   - {cracked_dir} (cracked tire images)")
    print(f"   - {normal_dir} (normal tire images)")
    print("   Based on the dataset's labeling scheme")

def main():
    """Main download function"""
    print("📥 Harvard Tire Damage Dataset Downloader")
    print("=" * 50)
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 Download Instructions:")
    print("1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FF1NQ3R")
    print("2. Request access to the dataset")
    print("3. Download the dataset files")
    print("4. Place the downloaded files in the data/raw/ directory")
    print("5. Run this script to organize the dataset")
    
    # Check if dataset files exist
    dataset_files = list(data_dir.glob("*"))
    if dataset_files:
        print(f"\n📁 Found {len(dataset_files)} files in data/raw/")
        print("Files found:")
        for file in dataset_files:
            print(f"   - {file.name}")
        
        # Ask user if they want to organize
        response = input("\n🤔 Do you want to organize the dataset? (y/n): ")
        if response.lower() == 'y':
            organize_dataset(data_dir)
    else:
        print("\n❌ No dataset files found in data/raw/")
        print("Please download the dataset first and place it in the data/raw/ directory")
    
    print("\n🎉 Dataset setup complete!")
    print("Next steps:")
    print("1. Organize images into cracked/ and normal/ folders")
    print("2. Run: python main_damage.py")

if __name__ == "__main__":
    main()
