# advanced_model.py - Enhanced tire damage detection with transfer learning
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json
import time
from datetime import datetime

class AdvancedTireDamageModel(nn.Module):
    """Advanced CNN model with transfer learning for tire damage classification"""
    
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True, dropout_rate=0.5):
        super(AdvancedTireDamageModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained backbone
        if model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove the original classifier
            
        elif model_name == 'efficientnet_b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'densenet121':
            if pretrained:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            else:
                self.backbone = models.densenet121(weights=None)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

class DamageSeverityModel(nn.Module):
    """Multi-class model for damage severity classification"""
    
    def __init__(self, num_classes=4, model_name='resnet50', pretrained=True):
        super(DamageSeverityModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained backbone
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Custom classifier for severity levels
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class AdvancedTrainer:
    """Advanced training class with multiple optimization strategies"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # Loss function with class weights
        if config.get('use_class_weights', False):
            class_weights = torch.FloatTensor(config['class_weights']).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-4)
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 1e-4)
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config.get('weight_decay', 1e-4)
            )
        
        # Learning rate scheduler
        if config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=2,
                verbose=True
            )
        elif config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.early_stop_patience = config.get('early_stop_patience', 5)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📊 Model: {self.model.model_name}")
        print(f"🔧 Optimizer: {self.config['optimizer']}")
        print(f"📈 Scheduler: {self.config['scheduler']}")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if self.config['scheduler'] == 'reduce_on_plateau':
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping based on validation accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"🛑 Early stopping at epoch {epoch+1} (no improvement in val_acc for {self.early_stop_patience} epochs)")
                    break
        
        print("✅ Training completed!")
        return self.history
    
    def save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path("models/checkpoint.pth")
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path("models/best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved (Val Acc: {val_acc:.2f}%)")

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return all_predictions, all_targets, all_probabilities
    
    def generate_report(self, predictions, targets, probabilities, class_names=['Normal', 'Cracked']):
        """Generate comprehensive evaluation report"""
        print("📊 Model Evaluation Report")
        print("=" * 50)
        
        # Basic metrics
        accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(targets)
        print(f"🎯 Overall Accuracy: {accuracy:.2%}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        print(f"\n🔢 Confusion Matrix:")
        print(cm)
        
        # ROC AUC
        if len(class_names) == 2:
            # Convert probabilities to numpy array if it's a list
            if isinstance(probabilities, list):
                probabilities = np.array(probabilities)
            roc_auc = roc_auc_score(targets, probabilities[:, 1])
            print(f"\n📈 ROC AUC Score: {roc_auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': classification_report(targets, predictions, target_names=class_names, output_dict=True)
        }
    
    def plot_training_history(self, history, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc')
        axes[0, 1].plot(history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 1].plot(loss_diff)
        axes[1, 1].set_title('Train-Val Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, targets, probabilities, class_names, save_path=None):
        """Plot ROC curve for binary classification"""
        if len(class_names) != 2:
            print("ROC curve only available for binary classification")
            return
        
        fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
        roc_auc = roc_auc_score(targets, probabilities[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 ROC curve saved to {save_path}")
        
        plt.show()

def get_advanced_transforms():
    """Get advanced data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
