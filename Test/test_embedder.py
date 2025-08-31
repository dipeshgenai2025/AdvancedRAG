# test_embedder
import os
import shutil
from Ingestion.pdf_parser import parse_pdf
from Ingestion.ocr import run_ocr_on_images
from Ingestion.image_BlipCaptioner import BlipCaptioner
from Ingestion.image_Captioner import Image_Captioner
from Embeddings.embedder import Embedder

def test_embedderModel():
    """
    Test the local embeddings generator.
    """
    temp_dir="TempData"
    result = parse_pdf("Test/Test.pdf", temp_dir)
    print("Extracted Text:", result["text"][:500])  # just first 500 chars

    # Initialize the embedder
    embedder = Embedder(model_path="./Models/EmbeddingModels/mpnet-base-v2", device=0)

    # Generate embeddings
    embeddings = embedder.encode(result["text"])
    print("Sample embedding (first 5 values):", embeddings[0][:500])
    print("Embedder test passed!")

    # Clean up temp directory after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nTemporary folder '{temp_dir}' deleted.")

if __name__ == "__main__":
    test_embedderModel()
