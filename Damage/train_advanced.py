# train_advanced.py - Advanced training script with experiment tracking
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Import our modules
from advanced_model import (
    AdvancedTireDamageModel, 
    AdvancedTrainer, 
    ModelEvaluator,
    get_advanced_transforms
)
from enhanced_main import create_data_loaders
from split_manager import set_seeds, create_fixed_splits, load_fixed_splits, get_class_weights

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Track training experiments and results"""
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)
        
        self.start_time = datetime.now()
        logger.info(f"🔬 Experiment started: {experiment_name}")
        logger.info(f"📁 Experiment directory: {self.experiment_dir}")
    
    def save_config(self, config: dict):
        """Save experiment configuration"""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"💾 Configuration saved to {config_path}")
    
    def save_model(self, model, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_acc': val_acc,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / "models" / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / "models" / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 Best model saved (Val Acc: {val_acc:.2f}%)")
    
    def save_results(self, results: dict):
        """Save experiment results"""
        results_path = self.experiment_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📊 Results saved to {results_path}")
    
    def save_plots(self, history: dict, cm: np.ndarray, class_names: list):
        """Save training plots"""
        # Training history
        self._plot_training_history(history)
        
        # Confusion matrix
        self._plot_confusion_matrix(cm, class_names)
        
        logger.info(f"📈 Plots saved to {self.experiment_dir / 'plots'}")
    
    def _plot_training_history(self, history: dict):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(history['learning_rates'], color='green')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(history['train_loss'], history['val_loss'])]
        axes[1, 1].plot(loss_diff, color='purple')
        axes[1, 1].set_title('Train-Val Loss Difference')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "training_history.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: list):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "plots" / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary(self, results: dict) -> str:
        """Generate experiment summary"""
        duration = datetime.now() - self.start_time
        
        summary = f"""
🔬 EXPERIMENT SUMMARY
{'=' * 50}
Experiment: {self.experiment_name}
Duration: {duration}
Directory: {self.experiment_dir}

📊 RESULTS
- Test Accuracy: {results.get('accuracy', 0):.2%}
- Best Val Accuracy: {results.get('best_val_acc', 0):.2%}
- Total Epochs: {results.get('total_epochs', 0)}
- Training Time: {results.get('training_time', 0):.2f}s

📁 FILES SAVED
- Models: {self.experiment_dir / 'models'}
- Plots: {self.experiment_dir / 'plots'}
- Config: {self.experiment_dir / 'config.json'}
- Results: {self.experiment_dir / 'results.json'}
- Logs: {self.experiment_dir / 'logs'}
"""
        return summary

def train_advanced_model(
    model_name: str = 'resnet50',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    data_dir: str = 'data',
    experiment_name: str = None,
    use_class_weights: bool = False,
    dropout_rate: float = 0.5,
    weight_decay: float = 1e-4,
    scheduler: str = 'reduce_on_plateau',
    early_stop_patience: int = 15
):
    """Train advanced damage detection model with full tracking"""
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_name}_{timestamp}"
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Set device - prioritize MPS for Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"🖥️ Using device: {device}")
    
    # Load or create fixed dataset splits
    splits_file = "splits.json"
    if Path(splits_file).exists():
        logger.info("📂 Loading existing splits...")
        splits = load_fixed_splits(splits_file)
    else:
        logger.info("📊 Creating new fixed splits...")
        splits = create_fixed_splits(data_dir, test_size=0.2, val_size=0.2, seed=42, splits_file=splits_file)
    
    # Extract data from splits
    train_data = (splits['train']['paths'], splits['train']['labels'])
    val_data = (splits['val']['paths'], splits['val']['labels'])
    test_data = (splits['test']['paths'], splits['test']['labels'])
    
    # Calculate class weights if needed
    class_weights = None
    if use_class_weights:
        class_weights = get_class_weights(splits, device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size
    )
    
    # Model configuration
    config = {
        'model_name': model_name,
        'optimizer': 'adamw',
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'scheduler': scheduler,
        'grad_clip': 1.0,
        'early_stop_patience': early_stop_patience,
        'use_class_weights': use_class_weights,
        'class_weights': class_weights.cpu().numpy().tolist() if class_weights is not None else None,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'data_dir': data_dir,
        'device': str(device)
    }
    
    # Save configuration
    tracker.save_config(config)
    
    # Initialize model
    logger.info(f"🤖 Initializing {model_name} model...")
    model = AdvancedTireDamageModel(
        num_classes=2,
        model_name=model_name,
        pretrained=True,
        dropout_rate=dropout_rate
    )
    model.to(device)
    
    logger.info(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, device, config)
    
    # Train model
    logger.info("🚀 Starting training...")
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, epochs)
    training_time = time.time() - start_time
    
    logger.info(f"⏱️ Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    logger.info("📊 Evaluating model...")
    evaluator = ModelEvaluator(model, device)
    predictions, targets, probabilities = evaluator.evaluate(test_loader)
    
    # Generate comprehensive report
    results = evaluator.generate_report(predictions, targets, probabilities)
    
    # Add additional results
    results['best_val_acc'] = max(history['val_acc'])
    results['total_epochs'] = len(history['train_loss'])
    results['training_time'] = training_time
    results['model_name'] = model_name
    results['experiment_name'] = experiment_name
    
    # Save results
    tracker.save_results(results)
    
    # Save model
    best_val_acc = max(history['val_acc'])
    tracker.save_model(model, len(history['train_loss']), best_val_acc, is_best=True)
    
    # Save plots
    cm = results['confusion_matrix']
    tracker.save_plots(history, cm, ['Normal', 'Cracked'])
    
    # Generate summary
    summary = tracker.generate_summary(results)
    logger.info(summary)
    
    # Save summary
    with open(tracker.experiment_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    return model, history, results, tracker

def compare_models(models: list, data_dir: str = 'data', epochs: int = 30):
    """Compare multiple models"""
    logger.info("🔬 Starting model comparison...")
    
    results = {}
    
    for model_name in models:
        logger.info(f"🚀 Training {model_name}...")
        
        try:
            model, history, result, tracker = train_advanced_model(
                model_name=model_name,
                epochs=epochs,
                data_dir=data_dir,
                experiment_name=f"comparison_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            results[model_name] = {
                'accuracy': result['accuracy'],
                'best_val_acc': result['best_val_acc'],
                'training_time': result['training_time'],
                'experiment_dir': tracker.experiment_dir
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Generate comparison report
    logger.info("📊 MODEL COMPARISON RESULTS")
    logger.info("=" * 50)
    
    for model_name, result in results.items():
        if 'error' in result:
            logger.info(f"{model_name}: ERROR - {result['error']}")
        else:
            logger.info(f"{model_name}:")
            logger.info(f"  Test Accuracy: {result['accuracy']:.2%}")
            logger.info(f"  Best Val Accuracy: {result['best_val_acc']:.2%}")
            logger.info(f"  Training Time: {result['training_time']:.2f}s")
            logger.info(f"  Experiment Dir: {result['experiment_dir']}")
    
    return results

def main():
    """Main training function with command line arguments"""
    parser = argparse.ArgumentParser(description='Advanced Tire Damage Detection Training')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'densenet121'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--class_weights', action='store_true', help='Use class weights')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple models
        models = ['resnet50', 'efficientnet_b0', 'densenet121']
        compare_models(models, args.data_dir, args.epochs)
    else:
        # Train single model
        train_advanced_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            data_dir=args.data_dir,
            experiment_name=args.experiment,
            use_class_weights=args.class_weights,
            dropout_rate=args.dropout,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            early_stop_patience=args.patience
        )

if __name__ == "__main__":
    main()
