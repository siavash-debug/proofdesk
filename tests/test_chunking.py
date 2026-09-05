import os

from ingestion.chunk import chunk_document
from ingestion.clean import clean_ocr_text
from ingestion.ocr import extract_raw_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _build_document(filename: str) -> dict:
    ocr_result = extract_raw_text(os.path.join(DATA_DIR, filename))
    cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])
    return {"filename": filename, **cleaned}


def test_certificate_produces_exactly_one_chunk():
    document = _build_document("cert_aws_sample.pdf")
    chunks = chunk_document(document)
    assert len(chunks) == 1


def test_resume_splits_into_one_chunk_per_section():
    document = _build_document("resume_ali_sample.pdf")
    chunks = chunk_document(document)
    assert len(chunks) == 4
    headers = {chunk["chunk_id"].split("::")[1] for chunk in chunks}
    assert headers == {"summary", "experience", "education", "skills"}


def test_every_chunk_has_non_empty_source_metadata():
    for filename in ["cert_aws_sample.pdf", "resume_ali_sample.pdf"]:
        document = _build_document(filename)
        for chunk in chunk_document(document):
            assert chunk["filename"] == filename
            assert chunk["chunk_id"]
            assert chunk["text"].strip() != ""
