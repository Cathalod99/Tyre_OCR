# Tire Damage Detection Pipeline

A high-performance tire damage detection system using EfficientNet-B0 with 96.07% accuracy.

## 🎯 Performance

- **Overall Accuracy**: 96.07%
- **Model Architecture**: EfficientNet-B0 (4.7M parameters)
- **Dataset**: 1,856 tire images (good/bad classification)
- **Training Time**: ~50 epochs with early stopping

### Per-Class Performance
- **Good Tires**: 93.85% precision, 97.60% recall
- **Bad Tires**: 98.00% precision, 94.84% recall

## 📁 Repository Structure

```
Damage/
├── advanced_model.py          # EfficientNet model architecture
├── damage_inference.py        # Main inference pipeline
├── evaluate_model.py          # Model evaluation script
├── train_advanced.py          # Training script
├── split_manager.py           # Dataset management
├── requirements.txt           # Dependencies
├── models/                    # Trained models
│   ├── best_model.pth        # 96% accuracy model (57MB)
│   └── checkpoint.pth        # Training checkpoint
├── data/                      # Dataset (gitignored)
│   ├── train/                # Training images
│   ├── val/                  # Validation images
│   └── test/                 # Test images
├── evaluation_results.json    # Performance metrics
├── evaluation_plots.png       # Performance visualizations
└── splits.json               # Dataset splits configuration
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference
```bash
# Single image
python damage_inference.py --model models/best_model.pth --image path/to/image.jpg --visualize

# Directory of images
python damage_inference.py --model models/best_model.pth --directory path/to/images/ --output results/
```

### 3. Evaluate Model
```bash
python evaluate_model.py --model models/best_model.pth
```

### 4. Train New Model
```bash
python train_advanced.py --model efficientnet_b0 --epochs 50 --batch_size 32
```

## 🔧 Key Features

### Advanced Model Architecture
- **EfficientNet-B0 backbone** with ImageNet pretrained weights
- **Custom classifier head** with dropout and batch normalization
- **Transfer learning** for optimal performance
- **Automatic architecture detection** from saved models

### Training Features
- **Multiple architectures**: EfficientNet-B0, ResNet50, DenseNet121
- **Advanced augmentations**: Random crops, flips, color jitter
- **Learning rate scheduling**: ReduceLROnPlateau, CosineAnnealing
- **Early stopping** with patience
- **Class weight balancing** for imbalanced datasets
- **Experiment tracking** with configs and results

### Inference Features
- **Batch processing** for multiple images
- **Confidence thresholds** for predictions
- **Visualization** with prediction overlays
- **Multiple output formats** (JSON, CSV, images)

## 📊 Dataset

The pipeline uses the Kaggle Tire Quality Classification dataset:
- **Total Images**: 1,856
- **Classes**: Good (828) vs Bad (1,028) tires
- **Splits**: 70% train, 15% validation, 15% test
- **Format**: JPEG images, 224x224 resolution
- **Augmentation**: Random crops, flips, color adjustments

## 🛠️ Model Training

### Training Configuration
```python
{
    "optimizer": "adamw",
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "scheduler": "reduce_on_plateau",
    "epochs": 50,
    "early_stop_patience": 10,
    "use_class_weights": true
}
```

### Supported Architectures
- **EfficientNet-B0**: Best performance (96.07% accuracy)
- **ResNet50**: Alternative architecture
- **DenseNet121**: Dense connectivity variant

## 📈 Evaluation Results

The model achieves excellent performance across all metrics:

```
Overall Accuracy: 96.07%

Per-Class Performance:
Good:  Precision=0.9385, Recall=0.9760, F1=0.9569
Bad:    Precision=0.9800, Recall=0.9484, F1=0.9639

Confusion Matrix:
           Pred Good    Pred Bad    
True Good  122          3           
True Bad   8            147         
```

## 🔍 Usage Examples

### Basic Inference
```python
from damage_inference import DamageDetector

# Load model
detector = DamageDetector('models/best_model.pth')

# Predict single image
result = detector.predict('path/to/tire_image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Batch Processing
```bash
python damage_inference.py \
    --model models/best_model.pth \
    --directory test_images/ \
    --output results/ \
    --confidence 0.8 \
    --visualize
```

### Custom Training
```bash
python train_advanced.py \
    --model efficientnet_b0 \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0005 \
    --experiment my_experiment \
    --class_weights
```

## 🎯 Production Ready

This pipeline is production-ready with:
- ✅ **High accuracy** (96.07%)
- ✅ **Fast inference** (< 1 second per image)
- ✅ **Robust error handling**
- ✅ **Comprehensive logging**
- ✅ **Model versioning**
- ✅ **Experiment tracking**
- ✅ **Clean, maintainable code**

## 📝 License

This project is part of a Masters thesis research project.

---

**Last Updated**: October 2024  
**Model Version**: EfficientNet-B0 v1.0  
**Accuracy**: 96.07%