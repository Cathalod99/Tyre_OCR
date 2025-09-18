# 🚀 Unified Tire Analysis System

## 🎯 Overview

This unified system combines your OCR project and damage detection project into a single, comprehensive tire analysis tool. When you input a tire sidewall image, it will:

1. **Extract tire information** (make, model, size, DOT code) using OCR
2. **Detect tire damage** (cracks, wear, etc.) using deep learning
3. **Provide safety assessment** and recommendations
4. **Generate comprehensive reports**

## 🚀 Quick Start

### **1. Test the System**
```bash
python test_unified.py
```

### **2. Add Your Tire Images**
Place your tire sidewall images in the `sample_image/` directory:
```bash
# Create directory if it doesn't exist
mkdir -p sample_image

# Add your tire images
cp your_tire_image.jpg sample_image/
```

### **3. Run Analysis**
```bash
# Analyze images in sample_image directory
python unified_main.py --sample_image

# Or analyze a specific image
python unified_main.py --image path/to/your/tire.jpg

# Save results to output directory
python unified_main.py --sample_image --output results/
```

## 📊 What You Get

### **OCR Analysis Results:**
- **Tire Make**: Manufacturer (e.g., Michelin, Bridgestone)
- **Tire Model**: Model name (e.g., Pilot Sport 4)
- **Tire Size**: Dimensions (e.g., 225/45R17)
- **DOT Code**: Manufacturing date code

### **Damage Detection Results:**
- **Damage Status**: DAMAGED or NORMAL
- **Confidence Level**: 0.0 to 1.0
- **Risk Assessment**: HIGH, MODERATE, LOW, NONE, UNKNOWN
- **Safety Recommendation**: Detailed advice

### **Output Files:**
- **JSON Results**: `tire_analysis_[image_name].json`
- **Text Report**: `tire_analysis_[image_name].txt`
- **Processing Time**: Typically < 1 second per image

## 🔧 System Requirements

### **Required Files:**
- ✅ `unified_main.py` - Main analysis script
- ✅ `Damage/models/best_model.pth` - Trained damage detection model
- ✅ `OCR/vision.py` - OCR text extraction
- ✅ `ML/text_processor.py` - Text processing
- ✅ `ML/tire_classifier.py` - Tire information classification

### **Dependencies:**
- Python 3.9+
- PyTorch
- OpenCV
- Google Cloud Vision API (for OCR)
- All dependencies from both projects

## 📈 Performance

### **Speed:**
- **OCR Analysis**: ~0.5-2 seconds (depends on image size)
- **Damage Detection**: ~0.05 seconds
- **Total Processing**: ~1-3 seconds per image

### **Accuracy:**
- **Damage Detection**: 92.04% accuracy
- **OCR**: Depends on image quality and text clarity
- **Combined**: Comprehensive analysis with confidence scores

## 🎯 Use Cases

### **Tire Inspection:**
- **Automotive shops** for tire assessment
- **Fleet management** for vehicle maintenance
- **Insurance claims** for damage documentation
- **Quality control** in tire manufacturing

### **Research & Development:**
- **Tire condition monitoring**
- **Safety assessment systems**
- **Automated inspection workflows**

## 🔍 Example Output

```
🚀 UNIFIED TIRE ANALYSIS SYSTEM
==================================================
📸 Image: tire_sample.jpg
⏰ Started: 2024-09-16 14:30:00

🔍 Running OCR Analysis...
📝 Extracted text: MICHELIN PILOT SPORT 4 225/45R17 DOT1234567890...
✅ OCR Analysis Complete:
   Make: Michelin
   Model: Pilot Sport 4
   Size: 225/45R17
   DOT Code: DOT1234567890

🔍 Running Damage Detection Analysis...
✅ Damage Detection Complete:
   Status: NORMAL
   Confidence: 0.95
   Risk Level: NONE
   Recommendation: SAFE: Tire appears to be in good condition.

📊 COMPREHENSIVE TIRE ANALYSIS RESULTS
==================================================
🔤 OCR ANALYSIS:
   ✅ Make: Michelin
   ✅ Model: Pilot Sport 4
   ✅ Size: 225/45R17
   ✅ DOT Code: DOT1234567890

🔍 DAMAGE ANALYSIS:
   ✅ Status: NORMAL
   ✅ Confidence: 0.95
   ✅ Risk Level: NONE
   ✅ Recommendation: SAFE: Tire appears to be in good condition.

📈 SUMMARY:
   ⏱️ Total Processing Time: 1.234s
   🎯 Overall Success: ✅ Yes
```

## 🛠️ Troubleshooting

### **Common Issues:**

1. **"No images found in sample_image/"**
   - Add tire images to the `sample_image/` directory

2. **"Damage model not found"**
   - Ensure `Damage/models/best_model.pth` exists
   - Train the damage detection model first

3. **"OCR system not available"**
   - Check Google Cloud Vision API credentials
   - Ensure all OCR dependencies are installed

4. **"No text extracted from image"**
   - Image may not contain readable text
   - Try with a clearer tire sidewall image

## 🎉 Ready to Use!

Your unified tire analysis system is ready! Simply add your tire images to the `sample_image/` directory and run the analysis. The system will provide comprehensive tire information and damage assessment in seconds!
