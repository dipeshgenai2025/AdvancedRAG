## qdrant_handler.py

"""
Qdrant operations.
Thread-safe: Each client created per request.
"""

from qdrant_client import QdrantClient # pyright: ignore[reportMissingImports]
from qdrant_client.models import Distance, VectorParams, PointStruct # pyright: ignore[reportMissingImports]
import uuid

class QdrantHandler:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "pdf_embeddings"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def create_collecion(self, vector_size: int):
        """Create collection if not exists"""
        if not self.client.collection_exists(self.collection_name) and vector_size is not None:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")

    def insert_embeddings(self, sentences, embeddings, pdf_id: str = "default_pdf", source: str = "pdf"):
        """
        sentences: list of text chunks (sentences or captions)
        embeddings: list of precomputed embeddings corresponding to each sentence
        pdf_id: identifier for the PDF
        source: "pdf" or "caption"
        """
        if len(sentences) != len(embeddings):
            raise ValueError("Length of sentences and embeddings must match.")

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

    def search(self, query_vector, top_k: int = 5):
        """
        query_vector: precomputed embedding of the query
        top_k: number of results
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results

    def delete_collection(self):
        """Danger: deletes the whole collection"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
