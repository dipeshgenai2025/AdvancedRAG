import os
import shutil
from Ingestion.pdf_parser import parse_pdf
from Ingestion.ocr import run_ocr_on_images
from Ingestion.image_BlipCaptioner import BlipCaptioner
from Ingestion.image_Captioner import Image_Captioner
from Embeddings.embedder import Embedder
from LLM.ollama_client import OllamaClient
from Utils.utils import format_text_by_sentences # pyright: ignore[reportMissingImports]
from Vectorstore.qdrant_handler import QdrantHandler


class RAGPipeline:
    def __init__(self, embedder_device=0, qdrant_url="http://localhost:6333", collection_name="pdf_embeddings"):
        self.embedder = Embedder(device=embedder_device)
        self.captioner = Image_Captioner("./Models/ImageCaptionModels/blip")
        self.qdrant_handler = QdrantHandler(url=qdrant_url, collection_name=collection_name)
        self.llm_client = OllamaClient(model="mistral:7b", url="http://localhost:11434")

    def ingest_pdf(self, pdf_path: str, temp_dir: str = "TempData"):
        """Parse PDF, run OCR, image captioning, generate embeddings, and insert into Qdrant"""
        os.makedirs(temp_dir, exist_ok=True)

        # 1. Parse PDF and add text, image, table and others in the result dictionary
        result = parse_pdf(pdf_path, temp_dir)

        # 2. Image captioning
        captions = self.captioner.caption(result["images"])
        result["caption"] = captions

        # 3. Combine text + captions
        #caption_texts = [f"Picture {idx}:{caption}" for idx, caption in enumerate(captions.values(), start=1)]
        caption_texts = [f"Picture {idx} : {caption}" for idx, caption in enumerate(result["caption"].values(), start=1)]
        combined_text = result["text"] + "\n" + "\n".join(caption_texts)

        # 4. Split text by sentences
        combined_text = format_text_by_sentences(combined_text)
        lines = combined_text.splitlines()

        # 5. Generate embeddings
        embeddings = [self.embedder.encode(line).squeeze(0).tolist() for line in lines]

        # 6. Store processed text in result
        result["formatted_text"] = lines
        result["embeddings"] = embeddings

        # 7. Create Qdrant collection and insert embeddings
        self.qdrant_handler.create_collecion(vector_size=len(result["embeddings"][0]))
        self.qdrant_handler.insert_embeddings(sentences=result["formatted_text"], embeddings=result["embeddings"], pdf_id=result["pdf_id"], source="pdf")

        # 8. Remove unnecessary keys to save memory
        del result["text"]
        del result["caption"]

        # 9. Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"\nTemporary folder '{temp_dir}' deleted.")

    def query(self, user_question: str, top_k: int = 10):
        """Query the Qdrant collection and return top-k relevant sentences"""
        query_vector = self.embedder.encode(user_question).squeeze(0).tolist()
        results = self.qdrant_handler.search(query_vector, top_k=top_k)

        if not results:
            return "No relevant information found."

        return [(hit.payload["text"], hit.score) for hit in results]

    def ask(self, user_question: str, top_k: int = 10):
        """Retrieve top-k context from Qdrant and generate answer using LLM"""
        retrieved = self.query(user_question, top_k=top_k)
        if isinstance(retrieved, str):  # No results
            return retrieved

        context = " ".join([text for text, score in retrieved])
        answer = self.llm_client.generate_answer(prompt=user_question, context=context)
        return answer