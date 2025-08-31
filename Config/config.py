## config.py

"""
Centralized configuration split into Qdrant and Ollama configs.
Thread-safe access ensured via frozen dataclasses.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "rag_collection"

@dataclass(frozen=True)
class OllamaConfig:
    host: str = "localhost"
    port: int = 11434
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "mistral"

@dataclass(frozen=True)
class AppConfig:
    qdrant: QdrantConfig = QdrantConfig()
    ollama: OllamaConfig = OllamaConfig()
    chunk_size: int = 500
    overlap: int = 50
