import os

import pytest
from dotenv import load_dotenv

from ingestion.clean import clean_ocr_text
from ingestion.ocr import extract_raw_text

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TEXT_BASED_PDFS = ["resume_ali_sample.pdf", "cert_aws_sample.pdf"]
SCANNED_PDF = "cert_scanned_sample.pdf"

requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment/.env",
)


@pytest.mark.parametrize("filename", TEXT_BASED_PDFS)
def test_text_based_pdf_skips_ocr(filename):
    result = extract_raw_text(os.path.join(DATA_DIR, filename))
    assert result["ocr_used"] is False
    assert result["filename"] == filename
    assert result["page_count"] > 0
    assert result["text"].strip() != ""


def test_scanned_pdf_requires_ocr():
    result = extract_raw_text(os.path.join(DATA_DIR, SCANNED_PDF))
    assert result["ocr_used"] is True
    assert result["filename"] == SCANNED_PDF
    assert result["page_count"] > 0


@requires_groq
def test_cleanup_fixes_known_ocr_misread_on_scanned_certificate():
    """Regression test for a known Tesseract misread on cert_scanned_sample.pdf.

    Raw OCR misreads the Roman numeral "I" in "Professional Scrum Master I"
    as a stray "|" (in the title line) and as "1" (in the "(PSM I)" line).
    The Groq cleanup step must restore "I" in both places rather than just
    dropping the stray character — this pins that behavior so a future
    prompt change can't silently regress it.
    """
    ocr_result = extract_raw_text(os.path.join(DATA_DIR, SCANNED_PDF))
    assert ocr_result["ocr_used"] is True
    assert "Master |" in ocr_result["text"]
    assert "PSM 1" in ocr_result["text"]

    cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])

    assert "Master I" in cleaned["cleaned_full_text"]
    assert "Master |" not in cleaned["cleaned_full_text"]
    assert "PSM I" in cleaned["cleaned_full_text"]

    assert cleaned["recipient_name"] == "Ali Sample"
    assert cleaned["issue_date"] == "September 2, 2024"
    assert cleaned["certificate_number"] == "PSM-2024-771402"


@requires_groq
def test_metadata_extraction_runs_for_text_layer_certificate():
    """Field extraction must run even when ocr_used=False.

    cert_aws_sample.pdf has a text layer, so no OCR correction is needed —
    but structured field extraction is a separate, independent step that
    must still run and populate recipient_name / certificate_number.
    """
    filename = "cert_aws_sample.pdf"
    ocr_result = extract_raw_text(os.path.join(DATA_DIR, filename))
    assert ocr_result["ocr_used"] is False

    cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])

    assert cleaned["recipient_name"] is not None
    assert cleaned["certificate_number"] is not None
    # Text-layer input must pass through unchanged, not be rewritten.
    assert cleaned["cleaned_full_text"] == ocr_result["text"]
