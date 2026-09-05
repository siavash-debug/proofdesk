import os
import shutil

import pytest
from dotenv import load_dotenv

from ingestion.chunk import chunk_document
from ingestion.clean import clean_ocr_text
from ingestion.embed import embed_and_store, get_collection
from ingestion.ocr import extract_raw_text

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_PDFS = ["cert_aws_sample.pdf", "resume_ali_sample.pdf", "cert_scanned_sample.pdf"]

requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment/.env",
)


@pytest.fixture
def test_collection(tmp_path):
    """A throwaway Chroma collection embedded with all 3 sample docs.

    Uses pytest's tmp_path (a fresh temp directory per test), never the
    real /chroma_db, so running tests can't pollute production data.
    """
    all_chunks = []
    for filename in SAMPLE_PDFS:
        ocr_result = extract_raw_text(os.path.join(DATA_DIR, filename))
        cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])
        document = {"filename": filename, **cleaned}
        all_chunks.extend(chunk_document(document))

    persist_dir = str(tmp_path / "chroma_test_db")
    embed_and_store(all_chunks, persist_dir=persist_dir)
    yield get_collection(persist_dir)

    shutil.rmtree(persist_dir, ignore_errors=True)


@requires_groq
def test_all_sample_chunks_are_embedded(test_collection):
    # cert_aws_sample.pdf -> 1 chunk, resume_ali_sample.pdf -> 4 chunks,
    # cert_scanned_sample.pdf -> 1 chunk = 6 total.
    assert test_collection.count() == 6


@requires_groq
@pytest.mark.parametrize(
    "query,expected_filename",
    [
        ("What AWS certification does Ali have?", "cert_aws_sample.pdf"),
        ("What is Ali's work experience with Kubernetes?", "resume_ali_sample.pdf"),
        ("What Scrum certification did Ali complete?", "cert_scanned_sample.pdf"),
    ],
)
def test_semantic_search_returns_correct_top_match(
    test_collection, query, expected_filename
):
    result = test_collection.query(query_texts=[query], n_results=1)
    top_match_filename = result["metadatas"][0][0]["filename"]
    assert top_match_filename == expected_filename
