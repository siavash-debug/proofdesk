"""Raw OCR extraction for scanned PDFs using Tesseract.

Prefers the PDF's own text layer (via pdfplumber) when one exists, since
that text is already clean and extracting it is far cheaper and more
reliable than OCR. Only falls back to Tesseract OCR (via pdf2image +
pytesseract) for pages with no extractable text layer at all.

Output is handed to `clean.py` for structuring before chunking.
"""

import os

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# A page is considered to have a usable text layer if pdfplumber extracts
# at least this many non-whitespace characters from it.
MIN_TEXT_LAYER_CHARS = 20


def _extract_text_layer(pdf_path: str) -> tuple[list[str], int]:
    """Return (per-page text, page_count) using the PDF's text layer.

    Any page below MIN_TEXT_LAYER_CHARS of extracted text is recorded as
    an empty string for that page.
    """
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text if len(text.strip()) >= MIN_TEXT_LAYER_CHARS else "")
    return pages, len(pages)


def _extract_via_ocr(pdf_path: str) -> list[str]:
    """Rasterize each page and run Tesseract OCR on it."""
    images = convert_from_path(pdf_path)
    return [pytesseract.image_to_string(image) for image in images]


def extract_raw_text(pdf_path: str) -> dict:
    """Extract text from a PDF, using the text layer when available.

    Returns a dict:
        {
            "filename": str,
            "page_count": int,
            "ocr_used": bool,
            "pages": list[str],   # raw text per page
            "text": str,          # all pages joined with double newlines
        }

    `ocr_used` is True only if Tesseract OCR was actually run (i.e. the
    PDF had no usable text layer on any page).
    """
    filename = os.path.basename(pdf_path)

    text_layer_pages, page_count = _extract_text_layer(pdf_path)
    has_text_layer = any(page.strip() for page in text_layer_pages)

    if has_text_layer:
        pages = text_layer_pages
        ocr_used = False
    else:
        pages = _extract_via_ocr(pdf_path)
        ocr_used = True

    return {
        "filename": filename,
        "page_count": page_count,
        "ocr_used": ocr_used,
        "pages": pages,
        "text": "\n\n".join(pages),
    }
