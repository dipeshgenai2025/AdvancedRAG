"""
OCR extraction from images.
Uses pytesseract + OpenCV.
"""

import pytesseract
from PIL import Image
from typing import List, Dict
import os
import logging

def run_ocr_on_images(image_paths: List[str]) -> Dict[str, str]:
    """
    Run OCR on a list of image files.

    Args:
        image_paths (List[str]): List of image file paths.

    Returns:
        Dict[str, str]: Dictionary mapping image_path -> extracted_text
    """
    if not isinstance(image_paths, list) or not all(isinstance(p, str) for p in image_paths):
        raise ValueError("image_paths must be a list of strings")
    if not image_paths:
        return {}

    ocr_results = {}

    for img_path in image_paths:
        if not os.path.exists(img_path):
            logging.warning(f"File not found: {img_path}")
            ocr_results[img_path] = "[File not found]"
            continue

        try:
            img = Image.open(img_path)
            text = pytesseract.image_to_string(img)
            ocr_results[img_path] = text.strip()
        except Exception as e:
            logging.error(f"Error during OCR on {img_path}: {e}")
            ocr_results[img_path] = f"[Error: {str(e)}]"

    return ocr_results
