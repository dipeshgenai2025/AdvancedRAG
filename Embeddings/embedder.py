## embedder.py
from transformers import AutoTokenizer, AutoModel # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
from typing import List, Union
import warnings
import threading
import logging

warnings.filterwarnings("ignore", category=UserWarning)

class Embedder:
    def __init__(self, model_path: str = "./Models/EmbeddingModels/mpnet-base-v2", device: int = 0):
        """
        Local embeddings generator
        :param model_path: Path to HuggingFace embedding model folder
        :param device: -1 for CPU, 0+ for GPU
        """
        self.lock = threading.Lock()
        try:
            self.device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts or a single text
        :param texts: Single text string or list of text strings
        :return: Tensor of shape (len(texts), embedding_dim) or (1, embedding_dim) if single text
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts must be a string or a list of strings")

        if not texts:
            raise ValueError("texts cannot be empty")

        with self.lock:
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)

                    outputs = self.model(**inputs)

                    # mean pooling (common for sentence embeddings)
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

                return embeddings.cpu()
            except Exception as e:
                logging.error(f"Error during encoding: {e}")
                raise RuntimeError(f"Encoding failed: {e}")
