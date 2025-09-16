# main_damage.py - Tire Damage Detection System
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class TireDamageDataset(Dataset):
    """Custom dataset for tire damage classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

class TireDamageCNN(nn.Module):
    """CNN model for tire damage classification"""
    
    def __init__(self, num_classes=2):
        super(TireDamageCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Assuming 224x224 input
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def load_dataset(data_dir):
    """Load the Harvard tire damage dataset"""
    print("🔄 Loading Harvard Tire Damage Dataset...")
    
    # Dataset paths - use training data for loading
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
    
    return image_paths, labels

def train_model(model, train_loader, val_loader, num_epochs=10):
    """Train the CNN model"""
    print("🚀 Starting model training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_accuracy = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    print("📊 Evaluating model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    accuracy = sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
    
    print(f"✅ Test Accuracy: {accuracy:.2%}")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['Normal', 'Cracked']))
    
    return all_predictions, all_targets

def main():
    """Main damage detection pipeline"""
    print("🔍 Tire Damage Detection System")
    print("=" * 40)
    
    # Start timing
    start_time = time.time()
    
    # Data directory
    data_dir = Path("data")
    
    # Check if dataset exists
    if not data_dir.exists():
        print("❌ Dataset not found!")
        print("Please download the Harvard Tire Damage Dataset to data/raw/")
        print("Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FF1NQ3R")
        return
    
    # Load dataset
    image_paths, labels = load_dataset(data_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found in dataset!")
        return
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split dataset
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"📊 Dataset Split:")
    print(f"   - Training: {len(train_paths)} images")
    print(f"   - Validation: {len(val_paths)} images")
    print(f"   - Test: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = TireDamageDataset(train_paths, train_labels, transform)
    val_dataset = TireDamageDataset(val_paths, val_labels, transform)
    test_dataset = TireDamageDataset(test_paths, test_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = TireDamageCNN(num_classes=2)
    print(f"🤖 Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    train_losses, val_accuracies = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Evaluate model
    predictions, targets = evaluate_model(model, test_loader)
    
    # Save model
    model_path = Path("models/tire_damage_model.pth")
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    print("🎉 Damage detection system complete!")

if __name__ == "__main__":
    main()
