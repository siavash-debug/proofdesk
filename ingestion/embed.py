"""Embed chunks and store them in the local vector store (ChromaDB).

Per CLAUDE.md's "no heavy vector DB" decision: this project uses ChromaDB
in local/persistent mode (a single directory on disk), not a managed
service — the corpus is small enough (< 20 documents) that a managed
vector DB would be pure operational overhead.

Uses ChromaDB's built-in default embedding function directly (Chroma's own
ONNX export of all-MiniLM-L6-v2, via ONNXMiniLM_L6_V2 — see CLAUDE.md's
"Embedding model in use" section) — no external embedding API call, and
no extra API key to manage, appropriate for this corpus size.

Embeds `chunk["embedding_text"]` rather than `chunk["text"]` (see
chunk.py): terse, list-style chunks embed poorly against natural-language
questions, so embedding_text carries a natural-language description
prefix for such chunks while `text` (what's actually shown/cited to the
user) stays unchanged. Since we need to embed different content than what
Chroma stores/returns as the document, embeddings are computed explicitly
here and passed to `upsert(embeddings=...)` rather than letting Chroma
auto-embed `documents=`.
"""

import os

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

# Same embedding function Chroma's DefaultEmbeddingFunction resolves to
# (see CLAUDE.md) — instantiated directly and reused across calls so we
# can embed `embedding_text` instead of the stored `text`, while querying
# (retrieval/search.py) still uses the collection's own default embedding
# function on the query text. Both resolve to the same underlying model,
# so the two embedding spaces stay consistent.
_EMBEDDING_FN = ONNXMiniLM_L6_V2()

# Metadata fields carried on each chunk (see chunk.py) that should be
# stored as Chroma metadata for filtering/attribution at query time.
CHUNK_METADATA_FIELDS = [
    "filename",
    "document_type",
    "title",
    "recipient_name",
    "issue_date",
    "certificate_number",
    "issuing_organization",
]


def get_collection(persist_dir: str = PERSIST_DIR):
    """Return the persistent Chroma collection, creating it if needed."""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_metadata(chunk: dict) -> dict:
    """Extract Chroma-safe metadata from a chunk (no None values, no nesting).

    Chroma metadata values must be str/int/float/bool — None is not
    allowed, so missing fields are stored as empty strings.
    """
    return {
        field: chunk.get(field) if chunk.get(field) is not None else ""
        for field in CHUNK_METADATA_FIELDS
    }


def embed_and_store(chunks: list[dict], persist_dir: str = PERSIST_DIR) -> None:
    """Embed each chunk's embedding_text and upsert it, with metadata, into Chroma.

    Uses `upsert` (keyed on chunk_id) rather than `add` so re-running
    ingestion on the same documents overwrites existing chunks instead of
    duplicating them.

    Embeddings are computed explicitly from `embedding_text` (falling
    back to `text` if a chunk has no embedding_text) and passed via
    `embeddings=`, so Chroma stores/returns `text` unchanged as the
    document while the vector reflects the enriched embedding_text.
    """
    if not chunks:
        return

    embedding_texts = [chunk.get("embedding_text") or chunk["text"] for chunk in chunks]
    embeddings = _EMBEDDING_FN(embedding_texts)

    collection = get_collection(persist_dir)
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[_chunk_metadata(chunk) for chunk in chunks],
    )
