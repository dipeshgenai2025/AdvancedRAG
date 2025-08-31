## embedder.py
from transformers import AutoTokenizer, AutoModel # pyright: ignore[reportMissingImports]
import torch # pyright: ignore[reportMissingImports]
from typing import List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class Embedder:
    def __init__(self, model_path: str = "./Models/EmbeddingModels/mpnet-base-v2", device: int = 0):
        """
        Local embeddings generator
        :param model_path: Path to HuggingFace embedding model folder
        :param device: -1 for CPU, 0+ for GPU
        """
        self.device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a list of texts
        :param texts: List of text strings
        :return: Tensor of shape (len(texts), embedding_dim)
        """
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
