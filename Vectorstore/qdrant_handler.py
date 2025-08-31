## qdrant_handler.py

"""
Qdrant operations.
Thread-safe: Each client created per request.
"""

from qdrant_client import QdrantClient # pyright: ignore[reportMissingImports]
from qdrant_client.models import Distance, VectorParams, PointStruct # pyright: ignore[reportMissingImports]
import uuid
import threading
import logging

class QdrantHandler:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "pdf_embeddings"):
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(collection_name, str) or not collection_name:
            raise ValueError("collection_name must be a non-empty string")

        try:
            self.client = QdrantClient(url=url)
            self.collection_name = collection_name
            self.lock = threading.Lock()
        except Exception as e:
            logging.error(f"Failed to initialize QdrantHandler: {e}")
            raise RuntimeError(f"QdrantHandler initialization failed: {e}")

    def create_collection(self, vector_size: int):
        """Create collection if not exists"""
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        with self.lock:
            try:
                if not self.client.collection_exists(self.collection_name):
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                    print(f"Created collection: {self.collection_name}")
            except Exception as e:
                logging.error(f"Error creating collection {self.collection_name}: {e}")
                raise RuntimeError(f"Collection creation failed: {e}")

    def insert_embeddings(self, sentences, embeddings, pdf_id: str = "default_pdf", source: str = "pdf"):
        """
        sentences: list of text chunks (sentences or captions)
        embeddings: list of precomputed embeddings corresponding to each sentence
        pdf_id: identifier for the PDF
        source: "pdf" or "caption"
        """
        if not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences):
            raise ValueError("sentences must be a list of strings")
        if not isinstance(embeddings, list) or not all(isinstance(e, list) and all(isinstance(v, (int, float)) for v in e) for e in embeddings):
            raise ValueError("embeddings must be a list of lists of numbers")
        if len(sentences) != len(embeddings):
            raise ValueError("Length of sentences and embeddings must match.")
        if not isinstance(pdf_id, str):
            raise ValueError("pdf_id must be a string")
        if not isinstance(source, str):
            raise ValueError("source must be a string")

        with self.lock:
            try:
                points = []
                for idx, (sentence, vector) in enumerate(zip(sentences, embeddings)):
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),  # unique ID for each vector
                            vector=vector,
                            payload={
                                "pdf_id": pdf_id,
                                "text": sentence,
                                "source": source
                            }
                        )
                    )

                self.client.upsert(collection_name=self.collection_name, points=points)
                print(f"Inserted {len(points)} embeddings into '{self.collection_name}'.")
            except Exception as e:
                logging.error(f"Error inserting embeddings into {self.collection_name}: {e}")
                raise RuntimeError(f"Embedding insertion failed: {e}")

    def search(self, query_vector, top_k: int = 5):
        """
        query_vector: precomputed embedding of the query
        top_k: number of results
        """
        if not isinstance(query_vector, list) or not all(isinstance(v, (int, float)) for v in query_vector):
            raise ValueError("query_vector must be a list of numbers")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with self.lock:
            try:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k
                )
                return results
            except Exception as e:
                logging.error(f"Error searching in {self.collection_name}: {e}")
                raise RuntimeError(f"Search failed: {e}")

    def delete_collection(self):
        """Danger: deletes the whole collection"""
        with self.lock:
            try:
                if self.client.collection_exists(self.collection_name):
                    self.client.delete_collection(self.collection_name)
                    print(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logging.error(f"Error deleting collection {self.collection_name}: {e}")
                raise RuntimeError(f"Collection deletion failed: {e}")
