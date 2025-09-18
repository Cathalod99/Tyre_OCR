# damage_inference.py - Damage detection inference and prediction utilities
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from advanced_model import AdvancedTireDamageModel, DamageSeverityModel, get_advanced_transforms

class DamageDetector:
    """Main damage detection class for inference"""
    
    def __init__(self, model_path: str, device: str = 'auto', confidence_threshold: float = 0.5):
        """
        Initialize damage detector
        
        Args:
            model_path: Path to trained model
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and configuration
        self.model, self.config = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        _, self.transform = get_advanced_transforms()
        
        print(f"🔍 Damage Detector initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self):
        """Load model and configuration from checkpoint"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
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
        
        return model, config
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    def predict_single(self, image_path: str) -> Dict:
        """Predict damage for a single image"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Convert to numpy
            confidence = confidence.cpu().item()
            predicted = predicted.cpu().item()
            probabilities = probabilities.cpu().numpy()[0]
            
            # Determine result
            is_damaged = predicted == 1
            is_confident = confidence >= self.confidence_threshold
            
            # Create result
            result = {
                'image_path': str(image_path),
                'prediction': 'damaged' if is_damaged else 'normal',
                'confidence': confidence,
                'is_confident': is_confident,
                'probabilities': {
                    'normal': float(probabilities[0]),
                    'damaged': float(probabilities[1])
                },
                'processing_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict damage for multiple images"""
        results = []
        
        print(f"🔍 Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(image_paths)}")
            
            result = self.predict_single(image_path)
            results.append(result)
        
        print(f"✅ Completed processing {len(image_paths)} images")
        return results
    
    def predict_directory(self, directory: str, pattern: str = "*.jpg") -> List[Dict]:
        """Predict damage for all images in a directory"""
        directory = Path(directory)
        image_paths = list(directory.glob(pattern))
        
        if not image_paths:
            print(f"⚠️ No images found in {directory} with pattern {pattern}")
            return []
        
        return self.predict_batch([str(p) for p in image_paths])
    
    def visualize_prediction(self, image_path: str, save_path: Optional[str] = None) -> None:
        """Visualize prediction result"""
        result = self.predict_single(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title(f"Original Image: {Path(image_path).name}")
        ax1.axis('off')
        
        # Show prediction
        prediction = result['prediction']
        confidence = result['confidence']
        color = 'red' if prediction == 'damaged' else 'green'
        
        ax2.bar(['Normal', 'Damaged'], result['probabilities'].values(), 
                color=['green', 'red'], alpha=0.7)
        ax2.set_title(f"Prediction: {prediction.title()}\nConfidence: {confidence:.3f}")
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add confidence threshold line
        ax2.axhline(y=self.confidence_threshold, color='orange', 
                   linestyle='--', label=f'Threshold: {self.confidence_threshold}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """Get statistics from prediction results"""
        if not results:
            return {}
        
        # Filter out errors
        valid_results = [r for r in results if 'error' not in r]
        error_count = len(results) - len(valid_results)
        
        if not valid_results:
            return {'error': 'No valid predictions'}
        
        # Calculate statistics
        total_images = len(valid_results)
        damaged_count = sum(1 for r in valid_results if r['prediction'] == 'damaged')
        normal_count = total_images - damaged_count
        confident_count = sum(1 for r in valid_results if r['is_confident'])
        
        avg_confidence = np.mean([r['confidence'] for r in valid_results])
        avg_processing_time = np.mean([r['processing_time'] for r in valid_results])
        
        # Confidence distribution
        confidences = [r['confidence'] for r in valid_results]
        high_confidence = sum(1 for c in confidences if c >= 0.8)
        medium_confidence = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.5)
        
        stats = {
            'total_images': total_images,
            'error_count': error_count,
            'damaged_count': damaged_count,
            'normal_count': normal_count,
            'damaged_percentage': (damaged_count / total_images) * 100,
            'confident_predictions': confident_count,
            'confidence_rate': (confident_count / total_images) * 100,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'confidence_distribution': {
                'high (≥0.8)': high_confidence,
                'medium (0.5-0.8)': medium_confidence,
                'low (<0.5)': low_confidence
            }
        }
        
        return stats

class DamageSeverityDetector:
    """Damage severity classification (mild, moderate, severe)"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == 'auto' else torch.device(device)
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        _, self.transform = get_advanced_transforms()
        
        # Severity levels
        self.severity_levels = ['normal', 'mild', 'moderate', 'severe']
        
        print(f"🔍 Damage Severity Detector initialized")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
    
    def _load_model(self):
        """Load severity model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        model_name = config.get('model_name', 'resnet50')
        
        model = DamageSeverityModel(
            num_classes=4,
            model_name=model_name,
            pretrained=False
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def predict_severity(self, image_path: str) -> Dict:
        """Predict damage severity"""
        start_time = time.time()
        
        try:
            # Preprocess image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            image = image.unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.cpu().item()
            predicted = predicted.cpu().item()
            probabilities = probabilities.cpu().numpy()[0]
            
            result = {
                'image_path': str(image_path),
                'severity': self.severity_levels[predicted],
                'confidence': confidence,
                'probabilities': {
                    level: float(prob) for level, prob in zip(self.severity_levels, probabilities)
                },
                'processing_time': time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'error': str(e),
                'processing_time': time.time() - start_time
            }

def create_damage_report(results: List[Dict], output_path: str = None) -> str:
    """Create a comprehensive damage report"""
    if not results:
        return "No results to report"
    
    # Filter valid results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        return "No valid results to report"
    
    # Generate report
    report = []
    report.append("🔍 TIRE DAMAGE DETECTION REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Images: {len(results)}")
    report.append(f"Valid Predictions: {len(valid_results)}")
    report.append("")
    
    # Summary statistics
    damaged_count = sum(1 for r in valid_results if r['prediction'] == 'damaged')
    normal_count = len(valid_results) - damaged_count
    avg_confidence = np.mean([r['confidence'] for r in valid_results])
    
    report.append("📊 SUMMARY")
    report.append("-" * 20)
    report.append(f"Damaged Tires: {damaged_count} ({damaged_count/len(valid_results)*100:.1f}%)")
    report.append(f"Normal Tires: {normal_count} ({normal_count/len(valid_results)*100:.1f}%)")
    report.append(f"Average Confidence: {avg_confidence:.3f}")
    report.append("")
    
    # Detailed results
    report.append("📋 DETAILED RESULTS")
    report.append("-" * 20)
    
    for i, result in enumerate(valid_results, 1):
        status = "✅" if result['is_confident'] else "⚠️"
        report.append(f"{i:3d}. {status} {Path(result['image_path']).name}")
        report.append(f"     Prediction: {result['prediction'].title()}")
        report.append(f"     Confidence: {result['confidence']:.3f}")
        report.append(f"     Processing Time: {result['processing_time']:.3f}s")
        report.append("")
    
    # Error summary
    error_results = [r for r in results if 'error' in r]
    if error_results:
        report.append("❌ ERRORS")
        report.append("-" * 10)
        for result in error_results:
            report.append(f"• {Path(result['image_path']).name}: {result['error']}")
        report.append("")
    
    report_text = "\n".join(report)
    
    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"📄 Report saved to {output_path}")
    
    return report_text

def main():
    """Example usage of damage detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tire Damage Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DamageDetector(args.model, confidence_threshold=args.confidence)
    
    # Process images
    if args.image:
        # Single image
        result = detector.predict_single(args.image)
        print(json.dumps(result, indent=2))
        
        if args.visualize:
            detector.visualize_prediction(args.image)
    
    elif args.directory:
        # Directory of images
        results = detector.predict_directory(args.directory)
        
        # Print statistics
        stats = detector.get_statistics(results)
        print("\n📊 STATISTICS")
        print("=" * 30)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Save results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            # Save JSON results
            with open(output_dir / 'damage_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save report
            report = create_damage_report(results, output_dir / 'damage_report.txt')
            print(f"\n{report}")
    
    else:
        print("Please specify either --image or --directory")

if __name__ == "__main__":
    main()
