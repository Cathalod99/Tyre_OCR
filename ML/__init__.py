from typing import Dict, Any
from .tire_classifier import best_brand_match, detect_model
from .dot_extractor import parse_tin
from .size_extractor import extract_size, extract_li_speed
from .text_processor import build_result

# Export the main functions
__all__ = [
    'best_brand_match',
    'detect_model',
    'parse_tin',
    'extract_size',
    'extract_li_speed',
    'build_result'
]

