# Tire Damage Detection System

Part 2 of the Tyre OCR Thesis Project - Automated tire damage detection using deep learning.

## 🎯 Overview

This module focuses on detecting and classifying tire damage using the Harvard Tire Damage Dataset, which contains 1,028 images of tires categorized into:
- **Cracked (oxidized)** tires
- **Normal** tires

## 📁 Project Structure

```
Damage/
├── README.md                 # This file
├── main_damage.py           # Main damage detection pipeline
├── requirements.txt         # Python dependencies
├── data/                    # Dataset storage
│   ├── raw/                # Original dataset
│   ├── processed/          # Preprocessed images
│   ├── train/              # Training split
│   ├── val/                # Validation split
│   └── test/               # Test split
├── models/                  # Trained models
├── utils/                   # Utility functions
└── notebooks/              # Jupyter notebooks for analysis
```

## 🚀 Quick Start

1. **Download the Harvard Dataset**:
   ```bash
   # Download from Harvard Dataverse
   # Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FF1NQ3R
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Damage Detection**:
   ```bash
   python main_damage.py
   ```

## 🔬 Methodology

### **Phase 1: Binary Classification**
- **Input**: Tire images
- **Output**: Cracked vs Normal classification
- **Model**: CNN-based classifier
- **Dataset**: Harvard Tire Damage Dataset (1,028 images)

### **Phase 2: Integration with OCR System**
- **Combined Pipeline**: OCR + Damage Detection
- **Unified Output**: Tire information + damage assessment
- **Safety Analysis**: Risk assessment based on damage severity

## 📊 Expected Results

- **Classification Accuracy**: >90% on test set
- **Processing Speed**: <1 second per image
- **Integration**: Seamless with existing OCR system

## 🔗 Integration with Part 1

This damage detection system integrates with the existing OCR system to provide:
1. **Tire Information**: Manufacturer, size, DOT code (Part 1)
2. **Damage Assessment**: Type, severity, safety recommendation (Part 2)
3. **Unified Analysis**: Complete tire health assessment

## 📚 References

- [Harvard Tire Damage Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FF1NQ3R)
- [CNNs for Tire Damage Detection](https://github.com/felitsch/CNNs-for-Tire-Damage-Detection)
- [Research on Tire Surface Damage Detection](https://www.mdpi.com/1424-8220/24/9/2778)
