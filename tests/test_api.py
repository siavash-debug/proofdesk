import os
import shutil

import pytest
from dotenv import load_dotenv

from api.main import answer_query
from api.schemas import ChatAnswer, NoMatchResponse
from ingestion.chunk import chunk_document
from ingestion.clean import clean_ocr_text
from ingestion.embed import embed_and_store, get_collection
from ingestion.ocr import extract_raw_text
from retrieval.search import search

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_PDFS = ["cert_aws_sample.pdf", "resume_ali_sample.pdf", "cert_scanned_sample.pdf"]

requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment/.env",
)


@pytest.fixture
def test_collection_dir(tmp_path):
    """Embed all 3 sample docs into a throwaway Chroma dir (never /chroma_db)."""
    all_chunks = []
    for filename in SAMPLE_PDFS:
        ocr_result = extract_raw_text(os.path.join(DATA_DIR, filename))
        cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])
        document = {"filename": filename, **cleaned}
        all_chunks.extend(chunk_document(document))

    persist_dir = str(tmp_path / "chroma_test_db")
    embed_and_store(all_chunks, persist_dir=persist_dir)
    yield persist_dir

    shutil.rmtree(persist_dir, ignore_errors=True)


@requires_groq
def test_generation_rejects_plausible_but_ungrounded_question(test_collection_dir):
    """A plausible-sounding question that retrieval alone does NOT catch.

    "Does Ali have an AWS Machine Learning Specialty certification?" is a
    variant of the calibration set's "plausible-sounding but not in any
    of our sample docs" case: it's about the exact right document
    (cert_aws_sample.pdf, distance ~0.63 — well under MIN_RELEVANCE_SCORE
    of 1.05, so retrieval definitely returns it as a strong match), but
    the actual certificate is "AWS Certified Solutions Architect —
    Associate", not "Machine Learning Specialty". Retrieval's threshold
    can't catch this (the retrieved chunk IS the closest match, and
    legitimately so) — only the second defense layer, the generation
    step's grounding check, can: it must notice the retrieved chunk
    doesn't actually answer the specific question asked and return
    NoMatchResponse instead of fabricating or implying a "yes".

    (Note: at MIN_RELEVANCE_SCORE=1.05, the original "PhD in Physics"
    calibration query is actually rejected by retrieval alone — its
    best-match distance, 1.0715, exceeds the 1.05 cutoff. This test uses
    a lower-distance plausible-but-wrong query instead, so it actually
    exercises the generation layer rather than being caught by retrieval
    first.)
    """
    query = "Does Ali have an AWS Machine Learning Specialty certification?"

    # Confirm retrieval alone lets this through (proves layer 2 is doing
    # the actual work here, not layer 1).
    retrieved_chunks = search(query, persist_dir=test_collection_dir)
    assert retrieved_chunks, (
        "expected retrieval to return candidates for this query at the "
        "current MIN_RELEVANCE_SCORE — if this fails, the calibration "
        "assumption behind this test has changed"
    )
    assert any(c["filename"] == "cert_aws_sample.pdf" for c in retrieved_chunks), (
        "expected the AWS certificate chunk to be retrieved (it's the "
        "closest match) — this is what makes it a real test of layer 2, "
        "not layer 1"
    )

    response = answer_query(query, persist_dir=test_collection_dir)

    assert isinstance(response, NoMatchResponse)


@requires_groq
def test_generation_answers_when_actually_grounded(test_collection_dir):
    """Sanity check: a genuinely answerable question still gets a real answer."""
    response = answer_query(
        "What AWS certification does Ali have?", persist_dir=test_collection_dir
    )

    assert isinstance(response, ChatAnswer)
    assert response.answer
    assert response.sources
    assert any(s.filename == "cert_aws_sample.pdf" for s in response.sources)
