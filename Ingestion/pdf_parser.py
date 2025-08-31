"""
PDF parsing utilities.
Extract text, tables, and images from PDFs.
Thread-safe since each request is independent.
"""

import os
import fitz # type: ignore
import pdfplumber # type: ignore
import uuid

def parse_pdf(pdf_path: str, temp_dir: str = "TempData"):
    """
    Parse PDF into text, tables, and images.
    Each PDF's images are stored in TempData/<pdf_basename_UUID>/
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_id = f"{base_name}_{uuid.uuid4().hex[:8]}"
    pdf_output_dir = os.path.join(temp_dir, pdf_id)
    os.makedirs(pdf_output_dir, exist_ok=True)

    text = ""
    tables = []
    images = []

    # -------- Extract text with PyMuPDF --------
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_path = os.path.join(pdf_output_dir, f"page{page.number}_img{img_index}.png")
            pix.save(img_path)
            images.append(img_path)
    doc.close()

    # -------- Extract tables with pdfplumber --------
    with pdfplumber.open(pdf_path) as plumber_pdf:
        for page in plumber_pdf.pages:
            tables.extend(page.extract_tables())

    return {
        "pdf_id": pdf_id,       # unique identifier for this PDF
        "text": text,
        "tables": tables,
        "images": images,
        "output_dir": pdf_output_dir
    }
