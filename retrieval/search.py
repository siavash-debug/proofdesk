"""Similarity search over the embedded document chunks.

No re-ranking stage in this phase — corpus size (< 20 documents) does not
justify the added complexity. Results are returned directly from the
vector store's similarity search, ordered by relevance, each carrying its
source metadata for attribution.

Relevance threshold contract (see CLAUDE.md "No-match handling"):
this module is the single enforcement point for the minimum-relevance
cutoff. `search()` must drop any candidate whose similarity score does not
meet MIN_RELEVANCE_SCORE before returning results. Callers (the API layer)
never see, and must never apply, their own threshold — if `search()`
returns an empty list, that IS the "no sufficiently relevant source"
signal, and the API must treat it as such rather than re-deciding
relevance itself.
"""

from ingestion.embed import PERSIST_DIR, get_collection

# Chroma returns a DISTANCE (lower = more similar), not a similarity
# score, despite this constant's name — "MIN_RELEVANCE_SCORE" is the
# established contract name (see CLAUDE.md), so it's kept for continuity,
# but the actual comparison below is `distance <= MIN_RELEVANCE_SCORE`.
#
# Calibrated against 6 manual queries (3 known-good, 3 known-bad) run
# against the 3-document sample corpus — see CLAUDE.md "No-match
# handling" for the full table. 1.05 was chosen to keep the weakest
# legitimate match observed (Kubernetes experience, distance 1.0292)
# while rejecting the most clearly irrelevant query observed (capital of
# France, distance 1.9873). This intentionally sits close to a borderline
# false-positive-shaped query (PhD in Physics, distance 1.0715) — see
# CLAUDE.md for why that gap is handled by a second layer, not this
# threshold.
MIN_RELEVANCE_SCORE = 1.05

# Tried reducing to 3 (corpus is only 6 chunks total) to address poor
# Contextual Relevancy scores from Phase 5 eval, but reverted: it didn't
# fix relevancy (the noise is CHUNK GRANULARITY — e.g. the Experience
# chunk mixes multiple jobs — not retrieval count) and it broke the
# previously-passing multi-fact synthesis case, which needs 3+ chunks
# from different documents to answer. See CLAUDE.md "Things to monitor".
DEFAULT_TOP_K = 5


def search(
    query: str, top_k: int = DEFAULT_TOP_K, persist_dir: str = PERSIST_DIR
) -> list[dict]:
    """Return up to top_k relevant chunks for the query, with metadata.

    Only chunks with distance <= MIN_RELEVANCE_SCORE are returned. If no
    chunk in the corpus meets the threshold (or the collection is empty),
    returns an empty list — this is the sole signal the API layer uses to
    decide "no matching source found."
    """
    collection = get_collection(persist_dir)
    if collection.count() == 0:
        return []

    n_results = min(top_k, collection.count())
    result = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if distance <= MIN_RELEVANCE_SCORE:
            chunks.append(
                {
                    **metadata,
                    "chunk_id": chunk_id,
                    "text": text,
                    "distance": distance,
                }
            )
    return chunks
