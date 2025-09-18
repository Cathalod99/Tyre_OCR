# integration.py - Integration with main OCR pipeline
import sys
import os
from pathlib import Path

# Add parent directory to path to import OCR modules
sys.path.append(str(Path(__file__).parent.parent))

from damage_inference import DamageDetector, DamageSeverityDetector, create_damage_report
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class TireAnalysisResult:
    """Complete tire analysis result combining OCR and damage detection"""
    # OCR Results
    tire_make: Optional[str] = None
    tire_model: Optional[str] = None
    tire_size: Optional[str] = None
    dot_code: Optional[str] = None
    
    # Damage Detection Results
    is_damaged: bool = False
    damage_confidence: float = 0.0
    damage_severity: Optional[str] = None
    damage_probabilities: Optional[Dict] = None
    
    # Safety Assessment
    safety_recommendation: Optional[str] = None
    risk_level: Optional[str] = None
    
    # Metadata
    processing_time: float = 0.0
    image_path: Optional[str] = None
    timestamp: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.timestamp is None:
            self.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

class IntegratedTireAnalyzer:
    """Integrated tire analysis combining OCR and damage detection"""
    
    def __init__(self, 
                 damage_model_path: str,
                 ocr_config: Optional[Dict] = None,
                 damage_confidence_threshold: float = 0.5):
        """
        Initialize integrated tire analyzer
        
        Args:
            damage_model_path: Path to trained damage detection model
            ocr_config: Configuration for OCR system
            damage_confidence_threshold: Confidence threshold for damage detection
        """
        self.damage_model_path = damage_model_path
        self.ocr_config = ocr_config or {}
        self.damage_confidence_threshold = damage_confidence_threshold
        
        # Initialize damage detector
        self.damage_detector = DamageDetector(
            model_path=damage_model_path,
            confidence_threshold=damage_confidence_threshold
        )
        
        # Initialize OCR components (import from parent directory)
        self.ocr_processor = None
        self._initialize_ocr()
        
        print("🔍 Integrated Tire Analyzer initialized")
        print(f"   Damage Model: {damage_model_path}")
        print(f"   OCR System: {'Available' if self.ocr_processor else 'Not Available'}")
    
    def _initialize_ocr(self):
        """Initialize OCR components from the main project"""
        try:
            # Import OCR modules
            from OCR.vision import OCRProcessor
            from ML.text_processor import TextProcessor
            from ML.tire_classifier import TireClassifier
            
            # Initialize OCR processor
            self.ocr_processor = OCRProcessor()
            
            # Initialize text processor
            self.text_processor = TextProcessor()
            
            # Initialize tire classifier
            self.tire_classifier = TireClassifier()
            
            print("✅ OCR system initialized successfully")
            
        except ImportError as e:
            print(f"⚠️ OCR system not available: {e}")
            self.ocr_processor = None
        except Exception as e:
            print(f"⚠️ Error initializing OCR system: {e}")
            self.ocr_processor = None
    
    def analyze_tire(self, image_path: str) -> TireAnalysisResult:
        """
        Perform complete tire analysis (OCR + damage detection)
        
        Args:
            image_path: Path to tire image
            
        Returns:
            TireAnalysisResult: Complete analysis result
        """
        start_time = time.time()
        result = TireAnalysisResult(image_path=image_path)
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                result.errors.append(f"Could not load image: {image_path}")
                return result
            
            # Perform OCR analysis
            if self.ocr_processor:
                try:
                    ocr_result = self._perform_ocr_analysis(image, image_path)
                    result.tire_make = ocr_result.get('make')
                    result.tire_model = ocr_result.get('model')
                    result.tire_size = ocr_result.get('size')
                    result.dot_code = ocr_result.get('dot_code')
                except Exception as e:
                    result.errors.append(f"OCR analysis failed: {str(e)}")
            else:
                result.errors.append("OCR system not available")
            
            # Perform damage detection
            try:
                damage_result = self.damage_detector.predict_single(image_path)
                
                if 'error' not in damage_result:
                    result.is_damaged = damage_result['prediction'] == 'damaged'
                    result.damage_confidence = damage_result['confidence']
                    result.damage_probabilities = damage_result['probabilities']
                else:
                    result.errors.append(f"Damage detection failed: {damage_result['error']}")
                    
            except Exception as e:
                result.errors.append(f"Damage detection failed: {str(e)}")
            
            # Perform safety assessment
            result.safety_recommendation = self._assess_safety(result)
            result.risk_level = self._assess_risk_level(result)
            
        except Exception as e:
            result.errors.append(f"Analysis failed: {str(e)}")
        
        finally:
            result.processing_time = time.time() - start_time
        
        return result
    
    def _perform_ocr_analysis(self, image: np.ndarray, image_path: str) -> Dict:
        """Perform OCR analysis on tire image"""
        try:
            # Extract text using OCR
            extracted_text = self.ocr_processor.extract_text(image)
            
            if not extracted_text:
                return {}
            
            # Process text to extract tire information
            processed_text = self.text_processor.process_text(extracted_text)
            
            # Classify tire information
            tire_info = self.tire_classifier.classify_tire_info(processed_text)
            
            return tire_info
            
        except Exception as e:
            print(f"OCR analysis error: {e}")
            return {}
    
    def _assess_safety(self, result: TireAnalysisResult) -> str:
        """Assess safety based on analysis results"""
        if result.is_damaged:
            if result.damage_confidence >= 0.8:
                return "HIGH RISK: Tire shows clear signs of damage. Immediate inspection recommended."
            elif result.damage_confidence >= 0.6:
                return "MODERATE RISK: Tire may have damage. Professional inspection recommended."
            else:
                return "LOW RISK: Possible minor damage. Monitor tire condition."
        else:
            if result.damage_confidence >= 0.8:
                return "SAFE: Tire appears to be in good condition."
            else:
                return "UNCERTAIN: Unable to determine tire condition with high confidence."
    
    def _assess_risk_level(self, result: TireAnalysisResult) -> str:
        """Assess risk level based on analysis results"""
        if result.is_damaged:
            if result.damage_confidence >= 0.8:
                return "HIGH"
            elif result.damage_confidence >= 0.6:
                return "MODERATE"
            else:
                return "LOW"
        else:
            if result.damage_confidence >= 0.8:
                return "NONE"
            else:
                return "UNKNOWN"
    
    def analyze_batch(self, image_paths: List[str]) -> List[TireAnalysisResult]:
        """Analyze multiple tire images"""
        results = []
        
        print(f"🔍 Analyzing {len(image_paths)} tire images...")
        
        for i, image_path in enumerate(image_paths):
            if i % 5 == 0:
                print(f"   Progress: {i}/{len(image_paths)}")
            
            result = self.analyze_tire(image_path)
            results.append(result)
        
        print(f"✅ Completed analysis of {len(image_paths)} images")
        return results
    
    def generate_comprehensive_report(self, results: List[TireAnalysisResult], 
                                    output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report"""
        if not results:
            return "No results to report"
        
        # Filter valid results
        valid_results = [r for r in results if not r.errors]
        error_count = len(results) - len(valid_results)
        
        # Generate report
        report = []
        report.append("🔍 COMPREHENSIVE TIRE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Images: {len(results)}")
        report.append(f"Successful Analyses: {len(valid_results)}")
        report.append(f"Failed Analyses: {error_count}")
        report.append("")
        
        # Summary statistics
        if valid_results:
            damaged_count = sum(1 for r in valid_results if r.is_damaged)
            high_risk_count = sum(1 for r in valid_results if r.risk_level == 'HIGH')
            avg_confidence = np.mean([r.damage_confidence for r in valid_results])
            avg_processing_time = np.mean([r.processing_time for r in valid_results])
            
            # OCR success rate
            ocr_success_count = sum(1 for r in valid_results if r.tire_make or r.tire_model or r.tire_size or r.dot_code)
            
            report.append("📊 SUMMARY STATISTICS")
            report.append("-" * 30)
            report.append(f"Damaged Tires: {damaged_count} ({damaged_count/len(valid_results)*100:.1f}%)")
            report.append(f"High Risk Tires: {high_risk_count} ({high_risk_count/len(valid_results)*100:.1f}%)")
            report.append(f"OCR Success Rate: {ocr_success_count/len(valid_results)*100:.1f}%")
            report.append(f"Average Confidence: {avg_confidence:.3f}")
            report.append(f"Average Processing Time: {avg_processing_time:.3f}s")
            report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 30)
        
        for i, result in enumerate(valid_results, 1):
            status = "🔴" if result.risk_level == "HIGH" else "🟡" if result.risk_level == "MODERATE" else "🟢"
            
            report.append(f"{i:3d}. {status} {Path(result.image_path).name}")
            
            # OCR Information
            if result.tire_make or result.tire_model or result.tire_size or result.dot_code:
                report.append(f"     OCR: {result.tire_make or 'N/A'} {result.tire_model or 'N/A'} "
                            f"{result.tire_size or 'N/A'} DOT:{result.dot_code or 'N/A'}")
            else:
                report.append(f"     OCR: No information extracted")
            
            # Damage Information
            damage_status = "DAMAGED" if result.is_damaged else "NORMAL"
            report.append(f"     Damage: {damage_status} (Confidence: {result.damage_confidence:.3f})")
            report.append(f"     Risk Level: {result.risk_level}")
            report.append(f"     Recommendation: {result.safety_recommendation}")
            report.append(f"     Processing Time: {result.processing_time:.3f}s")
            report.append("")
        
        # Error summary
        if error_count > 0:
            report.append("❌ ERRORS")
            report.append("-" * 10)
            for result in results:
                if result.errors:
                    report.append(f"• {Path(result.image_path).name}: {'; '.join(result.errors)}")
            report.append("")
        
        # Safety recommendations
        if valid_results:
            high_risk_tires = [r for r in valid_results if r.risk_level == 'HIGH']
            if high_risk_tires:
                report.append("⚠️ IMMEDIATE ATTENTION REQUIRED")
                report.append("-" * 35)
                for result in high_risk_tires:
                    report.append(f"• {Path(result.image_path).name}: {result.safety_recommendation}")
                report.append("")
        
        report_text = "\n".join(report)
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Comprehensive report saved to {output_path}")
        
        return report_text
    
    def save_results_json(self, results: List[TireAnalysisResult], output_path: str):
        """Save results as JSON"""
        # Convert results to serializable format
        json_results = []
        for result in results:
            json_result = {
                'image_path': result.image_path,
                'tire_make': result.tire_make,
                'tire_model': result.tire_model,
                'tire_size': result.tire_size,
                'dot_code': result.dot_code,
                'is_damaged': result.is_damaged,
                'damage_confidence': result.damage_confidence,
                'damage_severity': result.damage_severity,
                'damage_probabilities': result.damage_probabilities,
                'safety_recommendation': result.safety_recommendation,
                'risk_level': result.risk_level,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp,
                'errors': result.errors
            }
            json_results.append(json_result)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to {output_path}")

def main():
    """Example usage of integrated tire analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Tire Analysis (OCR + Damage Detection)')
    parser.add_argument('--damage_model', type=str, required=True, help='Path to damage detection model')
    parser.add_argument('--image', type=str, help='Path to single tire image')
    parser.add_argument('--directory', type=str, help='Path to directory of tire images')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=0.5, help='Damage detection confidence threshold')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = IntegratedTireAnalyzer(
        damage_model_path=args.damage_model,
        damage_confidence_threshold=args.confidence
    )
    
    # Analyze images
    if args.image:
        # Single image
        result = analyzer.analyze_tire(args.image)
        
        print("\n🔍 TIRE ANALYSIS RESULT")
        print("=" * 40)
        print(f"Image: {Path(result.image_path).name}")
        print(f"OCR Info: {result.tire_make} {result.tire_model} {result.tire_size} DOT:{result.dot_code}")
        print(f"Damage: {'DAMAGED' if result.is_damaged else 'NORMAL'} (Confidence: {result.damage_confidence:.3f})")
        print(f"Risk Level: {result.risk_level}")
        print(f"Recommendation: {result.safety_recommendation}")
        print(f"Processing Time: {result.processing_time:.3f}s")
        
        if result.errors:
            print(f"Errors: {'; '.join(result.errors)}")
    
    elif args.directory:
        # Directory of images
        image_paths = list(Path(args.directory).glob("*.jpg"))
        results = analyzer.analyze_batch([str(p) for p in image_paths])
        
        # Generate report
        report = analyzer.generate_comprehensive_report(results)
        print(f"\n{report}")
        
        # Save results
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            
            # Save JSON results
            analyzer.save_results_json(results, output_dir / 'tire_analysis_results.json')
            
            # Save report
            with open(output_dir / 'tire_analysis_report.txt', 'w') as f:
                f.write(report)
    
    else:
        print("Please specify either --image or --directory")

if __name__ == "__main__":
    main()
