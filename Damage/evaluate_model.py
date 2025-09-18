# evaluate_model.py - Evaluate best model checkpoint on test set
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import argparse

# Import our modules
from advanced_model import AdvancedTireDamageModel, ModelEvaluator, get_advanced_transforms
from enhanced_main import load_dataset, create_data_loaders

def evaluate_best_model(model_path: str, data_dir: str = 'data', batch_size: int = 32):
    """Evaluate the best model checkpoint on test set"""
    
    print("🔍 MODEL EVALUATION ON TEST SET")
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
    print("📊 Loading test dataset...")
    train_data, val_data, test_data = load_dataset(data_dir)
    
    if len(test_data[0]) == 0:
        print("❌ No test data found!")
        return None
    
    print(f"✅ Test dataset: {len(test_data[0])} images")
    
    # Create test data loader
    _, val_transform = get_advanced_transforms()
    from enhanced_main import TireDamageDataset
    
    test_dataset = TireDamageDataset(test_data[0], test_data[1], val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    print(f"🤖 Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'resnet50')
    
    # Initialize model
    model = AdvancedTireDamageModel(
        num_classes=2,
        model_name=model_name,
        pretrained=False,
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model_name}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate model
    print("\n🔍 Evaluating on test set...")
    evaluator = ModelEvaluator(model, device)
    predictions, targets, probabilities = evaluator.evaluate(test_loader)
    
    # Calculate detailed metrics
    print("\n📊 DETAILED EVALUATION RESULTS")
    print("=" * 50)
    
    # Overall accuracy
    accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(targets)
    print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, labels=[0, 1]
    )
    
    class_names = ['Normal', 'Cracked']
    print(f"\n📋 Per-Class Performance:")
    print("-" * 40)
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 40)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    print(f"\n🔢 Confusion Matrix:")
    print("-" * 20)
    print(f"{'':<10} {'Pred Normal':<12} {'Pred Cracked':<12}")
    print("-" * 20)
    print(f"{'True Normal':<10} {cm[0,0]:<12} {cm[0,1]:<12}")
    print(f"{'True Cracked':<10} {cm[1,0]:<12} {cm[1,1]:<12}")
    
    # Analysis
    print(f"\n🔍 CLASS PERFORMANCE ANALYSIS:")
    print("-" * 40)
    
    # Normal class analysis
    normal_precision = precision[0]
    normal_recall = recall[0]
    normal_f1 = f1[0]
    
    # Cracked class analysis
    cracked_precision = precision[1]
    cracked_recall = recall[1]
    cracked_f1 = f1[1]
    
    print(f"Normal Class:")
    print(f"  Precision: {normal_precision:.4f} - {'✅ Good' if normal_precision > 0.9 else '⚠️ Needs improvement'}")
    print(f"  Recall: {normal_recall:.4f} - {'✅ Good' if normal_recall > 0.9 else '⚠️ Needs improvement'}")
    print(f"  F1-Score: {normal_f1:.4f} - {'✅ Good' if normal_f1 > 0.9 else '⚠️ Needs improvement'}")
    
    print(f"\nCracked Class:")
    print(f"  Precision: {cracked_precision:.4f} - {'✅ Good' if cracked_precision > 0.9 else '⚠️ Needs improvement'}")
    print(f"  Recall: {cracked_recall:.4f} - {'✅ Good' if cracked_recall > 0.9 else '⚠️ Needs improvement'}")
    print(f"  F1-Score: {cracked_f1:.4f} - {'✅ Good' if cracked_f1 > 0.9 else '⚠️ Needs improvement'}")
    
    # Identify which class is weaker
    print(f"\n⚖️ CLASS COMPARISON:")
    print("-" * 20)
    
    if cracked_f1 < normal_f1 - 0.05:  # 5% threshold
        print("🔴 Cracked class is WEAKER - needs attention!")
        print("💡 Recommendations:")
        print("   - Increase class weights for cracked samples")
        print("   - Add more data augmentation for cracked samples")
        print("   - Consider focal loss for imbalanced classes")
        print("   - Collect more cracked tire images")
    elif normal_f1 < cracked_f1 - 0.05:
        print("🔴 Normal class is WEAKER - needs attention!")
        print("💡 Recommendations:")
        print("   - Increase class weights for normal samples")
        print("   - Add more data augmentation for normal samples")
    else:
        print("✅ Both classes performing similarly - good balance!")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names))
    
    # Save results
    results = {
        'overall_accuracy': float(accuracy),
        'per_class_metrics': {
            'normal': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0]),
                'support': int(support[0])
            },
            'cracked': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1]),
                'support': int(support[1])
            }
        },
        'confusion_matrix': cm.tolist(),
        'model_path': model_path,
        'test_samples': len(targets)
    }
    
    # Save to file
    results_path = Path("evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {results_path}")
    
    # Create visualization
    create_evaluation_plots(cm, class_names, precision, recall, f1)
    
    return results

def create_evaluation_plots(cm, class_names, precision, recall, f1):
    """Create evaluation visualization plots"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # Per-class Precision
    bars1 = axes[1].bar(class_names, precision, color=['green', 'red'], alpha=0.7)
    axes[1].set_title('Per-Class Precision')
    axes[1].set_ylabel('Precision')
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(precision):
        axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Per-class Recall
    bars2 = axes[2].bar(class_names, recall, color=['green', 'red'], alpha=0.7)
    axes[2].set_title('Per-Class Recall')
    axes[2].set_ylabel('Recall')
    axes[2].set_ylim(0, 1)
    for i, v in enumerate(recall):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    print("📊 Evaluation plots saved to evaluation_plots.png")
    plt.show()

def find_best_model():
    """Find the best model checkpoint"""
    
    # Look for best model in experiments directory
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        # Find the most recent experiment
        experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        if experiment_dirs:
            latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
            best_model_path = latest_experiment / "models" / "best_model.pth"
            if best_model_path.exists():
                return str(best_model_path)
    
    # Look for best model in models directory
    models_dir = Path("models")
    if models_dir.exists():
        best_model_path = models_dir / "best_model.pth"
        if best_model_path.exists():
            return str(best_model_path)
    
    # Look for any .pth file in models directory
    if models_dir.exists():
        pth_files = list(models_dir.glob("*.pth"))
        if pth_files:
            return str(pth_files[0])
    
    return None

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate best model on test set')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model:
        model_path = args.model
    else:
        model_path = find_best_model()
        if not model_path:
            print("❌ No model found! Please specify --model or ensure models exist.")
            return
        print(f"🔍 Found model: {model_path}")
    
    # Evaluate model
    results = evaluate_best_model(model_path, args.data_dir, args.batch_size)
    
    if results:
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Overall accuracy: {results['overall_accuracy']:.4f}")
        print(f"📋 Check evaluation_results.json for detailed metrics")

if __name__ == "__main__":
    main()
