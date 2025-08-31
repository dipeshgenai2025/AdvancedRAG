# test_pdf_parser.py
import os
import shutil
from Ingestion.pdf_parser import parse_pdf
from Ingestion.ocr import run_ocr_on_images
from Ingestion.image_BlipCaptioner import BlipCaptioner
from Ingestion.image_Captioner import Image_Captioner

def test_parse_pdf():
    pdf_path = "Test/Test.pdf"  # Ensure this test PDF exists
    temp_dir="TempData"
    result = parse_pdf(pdf_path, temp_dir)

    assert "pdf_id" in result
    assert isinstance(result["text"], str)
    assert isinstance(result["tables"], list)
    assert isinstance(result["images"], list)
    assert os.path.exists(result["output_dir"])

    print("\n--- Extracted Text ---\n")
    print(result["text"][:1000])  # print first 1000 chars

    print("\n--- Tables ---\n")
    for table in result["tables"]:
        for row in table:
            print(row)
        print("-" * 50)

    print("\n--- Images ---\n")
    for img in result["images"]:
        print(img)

    print("PDF parsing test passed.")

    # Clean up temp directory after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nTemporary folder '{temp_dir}' deleted.")

def test_ocr_on_images():
    temp_dir="TempData"
    result = parse_pdf("Test/Test.pdf", temp_dir)
    print("Extracted Text:", result["text"][:500])  # just first 500 chars

    ocr_texts = run_ocr_on_images(result["images"])
    for img, text in ocr_texts.items():
        print(f"\nOCR from {img}:\n{text}\n")

    print("OCR test passed.")

    # Clean up temp directory after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nTemporary folder '{temp_dir}' deleted.")

def test_image_BlipCaptioner():
    temp_dir="TempData"
    result = parse_pdf("Test/Test.pdf", temp_dir)
    #print("Extracted Text:", result["text"][:500])  # just first 500 chars

    captioner = BlipCaptioner("./Models/ImageCaptionModels/blip")
    caption = captioner.caption_images(result["images"])
    print("Caption:", caption)
    print("Image understanding test passed.")

    # Clean up temp directory after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nTemporary folder '{temp_dir}' deleted.")

def test_Image_Captioner():
    temp_dir="TempData"
    result = parse_pdf("Test/Test.pdf", temp_dir)
    #print("Extracted Text:", result["text"][:500])  # just first 500 chars

    captioner = Image_Captioner("./Models/ImageCaptionModels/blip")
    caption = captioner.caption(result["images"])
    print("Caption:", caption)
    print("Image understanding test passed.")

    # Clean up temp directory after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nTemporary folder '{temp_dir}' deleted.")

if __name__ == "__main__":
    #test_parse_pdf()
    #test_ocr_on_images()
    #test_Image_Captioner()
    test_image_BlipCaptioner()
