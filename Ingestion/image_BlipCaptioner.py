"""
Image understanding via local captioning (BLIP).
- No network calls required after first install of models.
- Thread-safe lazy loader with a lock.
- Works on CPU or GPU (auto-detect).
"""

from typing import List, Dict
from PIL import Image
import torch # pyright: ignore[reportMissingImports]
import threading

from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore

class BlipCaptioner:
    """
    Thread-safe, lazy-initialized BLIP captioner.
    Usage:
        cap = BlipCaptioner(model_name="Salesforce/blip-image-captioning-base")
        captions = cap.caption_images(["/path/img1.png", "/path/img2.png"])
    """
    _init_lock = threading.Lock()

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: str | None = None):
        self.model_name = model_name
        self._processor = None
        
        self._model = None
        # Device selection (CPU by default; uses CUDA if available or MPS on Apple)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():  # Apple Silicon
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def _ensure_loaded(self):
        # Lazy, thread-safe loading
        if self._processor is None or self._model is None:
            with BlipCaptioner._init_lock:
                if self._processor is None or self._model is None:
                    self._processor = BlipProcessor.from_pretrained(self.model_name, local_files_only=True, use_fast=True)
                    self._model = BlipForConditionalGeneration.from_pretrained(self.model_name, local_files_only=True)
                    self._model.to(self.device)
                    self._model.eval()

    def caption_images(
        self, image_paths: List[str],
        max_new_tokens: int = 80,
        num_beams: int = 10,
        repetition_penalty: float = 1.8) -> Dict[str, str]:

        """
        max_new_tokens - allow longer captions
        num_beams - beam search for higher quality
        Returns: dict[image_path] -> caption
        Robust to missing/corrupt files (caption will be an error string).
        """
        self._ensure_loaded()
        results: Dict[str, str] = {}

        for p in image_paths:
            try:
                image = Image.open(p).convert("RGB")
            except Exception as e:
                results[p] = f"[error opening image: {e}]"
                continue

            try:
                inputs = self._processor(images=image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty)

                caption = self._processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                results[p] = caption

            except Exception as e:
                results[p] = f"[captioning error: {e}]"

        return results
