"""
OCR extraction from images.
Uses pytesseract + OpenCV.
"""

import pytesseract
from PIL import Image
from typing import List, Dict
import os

def run_ocr_on_images(image_paths: List[str]) -> Dict[str, str]:
    """
    Run OCR on a list of image files.
    
    Args:
        image_paths (List[str]): List of image file paths.
    
    Returns:
        Dict[str, str]: Dictionary mapping image_path -> extracted_text
    """
    ocr_results = {}

    for img_path in image_paths:
        if not os.path.exists(img_path):
            ocr_results[img_path] = "[File not found]"
            continue

        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            ocr_results[img_path] = text.strip()
        except Exception as e:
            ocr_results[img_path] = f"[Error: {str(e)}]"

    return ocr_results
