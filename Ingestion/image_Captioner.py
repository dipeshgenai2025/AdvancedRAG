# Ingestion/captioner.py
from transformers import pipeline # type: ignore

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

        self.pipe = pipeline("image-to-text", model=model_name, device=device, use_fast=True)
        #self.pipe = pipeline("image-to-text", model=model_name, device=device, trust_remote_code=True, use_fast=True)

    #def caption(self, image_paths: list[str]) -> dict:
    def caption(self, image_paths, prompt="a detailed description of this image"):
        """
        Generate captions for a list of image paths with tuned decoding parameters.
        """
        captions = {}
        for img_path in image_paths:
            try:
                result = self.pipe(
                    img_path,
                    max_new_tokens=80,
                )
                captions[img_path] = result[0]['generated_text']
            except Exception as e:
                captions[img_path] = f"Error: {str(e)}"
        return captions
