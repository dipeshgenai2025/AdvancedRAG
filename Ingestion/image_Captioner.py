# Ingestion/captioner.py
from transformers import pipeline # type: ignore
from typing import List, Dict
import threading
import logging

class Image_Captioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: int = 0):
        """
        Initialize the image captioning model.
        :param model_name: HuggingFace model name (BLIP, BLIP-2, Florence-2, PaliGemma, etc.)
        :param device: -1 for CPU, 0+ for GPU device
        """
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

        self.lock = threading.Lock()
        try:
            self.pipe = pipeline("image-to-text", model=model_name, device=device, use_fast=True)
            #self.pipe = pipeline("image-to-text", model=model_name, device=device, trust_remote_code=True, use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def caption(self, image_paths: List[str], prompt: str = "a detailed description of this image") -> Dict[str, str]:
        """
        Generate captions for a list of image paths with tuned decoding parameters.
        """
        if not isinstance(image_paths, list) or not all(isinstance(p, str) for p in image_paths):
            raise ValueError("image_paths must be a list of strings")
        if not image_paths:
            return {}

        captions = {}
        with self.lock:
            for img_path in image_paths:
                try:
                    result = self.pipe(
                        img_path,
                        max_new_tokens=80,
                    )
                    captions[img_path] = result[0]['generated_text']
                except Exception as e:
                    logging.error(f"Error captioning image {img_path}: {e}")
                    captions[img_path] = f"Error: {str(e)}"
        return captions
