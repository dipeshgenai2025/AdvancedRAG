## main.py
"""
FastAPI entry point.
Coordinates ingestion, retrieval, and answering.
"""
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from fastapi import FastAPI, UploadFile
# from config import AppConfig

# app = FastAPI()
# config = AppConfig()

# @app.get("/")
# def health_check():
#     return {"status": "ok"}

# @app.post("/upload_pdf")
# async def upload_pdf(file: UploadFile):
#     TODO: call ingestion pipeline
#     return {"filename": file.filename}

# @app.post("/query")
# async def query_llm(query: str):
#     TODO: embed query, retrieve docs, call Ollama
#     return {"query": query, "answer": "TBD"}

from RAG_Pipeline.RAG_Pipeline import RAGPipeline # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    try:
        pdf_path = "Test/Test.pdf"
        pipeline = RAGPipeline(embedder_device=0, collection_name="pdf_embeddings")

        # Ingest PDF and store embeddings
        # TODO : NOTE: Comment this line after first run, Qdrant DB will already been updated
        logging.info("Starting PDF ingestion...")
        result = pipeline.ingest_pdf(pdf_path, temp_dir="TempData")
        logging.info("PDF ingestion completed.")

        # Query example
        # user_question = "Who is Burns?"
        # user_question = "What is in picture 3 ?"
        user_question = "What is the document all about ?"

        logging.info(f"Querying: {user_question}")
        answer = pipeline.ask(user_question)
        print("Answer:", answer)
        logging.info("Query completed.")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
