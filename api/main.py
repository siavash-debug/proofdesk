"""Backend API: serves the frontend and the chat endpoint.

Orchestrates: retrieval.search -> branch on result -> LLM answer
generation -> response with mandatory source citations (filename,
section/heading).

Exposes POST /chat as the HTTP endpoint the frontend (/frontend) calls.

No-match contract (see CLAUDE.md "No-match handling" and api/schemas.py):
the chat endpoint must branch on retrieval.search()'s result before
calling the generation LLM. An empty result returns NoMatchResponse
directly; the LLM is never called in that case.

Two-layer defense against fabricated/loosely-related answers (see
CLAUDE.md "No-match handling"):
1. retrieval.search()'s MIN_RELEVANCE_SCORE (favors recall — a coarse
   embedding-distance filter, deliberately loose).
2. This module's generation step (favors precision) — the prompt sent to
   the answer-generation LLM explicitly requires it to verify a retrieved
   chunk actually CONTAINS the answer, not merely that it's topically
   related, and to fall back to NoMatchResponse if none do. This catches
   the borderline case retrieval's threshold intentionally lets through
   (e.g. an AWS certificate chunk retrieved for a "PhD in Physics"
   question — topically in the same "credentials" space, but it doesn't
   answer the question).
"""

import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

from api.schemas import ChatAnswer, Citation, NoMatchResponse
from ingestion.embed import PERSIST_DIR
from retrieval.search import DEFAULT_TOP_K, search

# Loaded here (not just in tests) because this module is the actual
# server entrypoint (`uvicorn api.main:app`) — GROQ_API_KEY/GROQ_MODEL
# must be available in-process, not just to whatever shell happens to
# have sourced .env manually. Does nothing if .env doesn't exist or vars
# are already set in the environment.
load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

GENERATION_SYSTEM_PROMPT = """You are answering questions about a small set of \
certificate/resume documents, using ONLY the context chunks provided below. \
Your audience includes recruiters evaluating a candidate, so a wrong or \
fabricated answer is worse than saying the information isn't available.

Before answering, verify that at least one context chunk actually CONTAINS \
information that answers the question — not merely that it is topically \
related or about a similar subject. For example, a chunk about an AWS \
certification is topically related to "credentials" in general, but it does \
NOT answer a question about a PhD in Physics, and must not be used to \
fabricate or imply an answer to that question.

Respond with JSON only, with exactly these keys:
- "found": true if and only if a context chunk actually contains the \
answer to the question; false otherwise (including when chunks are only \
topically related but don't contain the answer).
- "answer": the answer text if found is true, else null. Base this only \
on the provided context — never use outside knowledge.
- "source_chunk_ids": a list of the chunk_id values (from the context) \
that the answer is actually grounded in. Empty list if found is false."""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[chunk_id: {chunk['chunk_id']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> ChatAnswer | NoMatchResponse:
    """Generate an answer grounded in `chunks`, or NoMatchResponse.

    This is the second defense layer: even though `chunks` already passed
    retrieval's MIN_RELEVANCE_SCORE, the LLM must independently verify at
    least one chunk actually contains the answer before producing one.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context chunks:\n\n{_build_context(chunks)}\n\n"
                f"Question: {query}",
            },
        ],
    )
    result = json.loads(response.choices[0].message.content)

    if not result.get("found") or not result.get("answer"):
        return NoMatchResponse()

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    source_ids = result.get("source_chunk_ids") or list(chunks_by_id.keys())
    sources = [
        Citation(
            filename=chunks_by_id[chunk_id]["filename"],
            chunk_id=chunk_id,
            title=chunks_by_id[chunk_id].get("title") or None,
        )
        for chunk_id in source_ids
        if chunk_id in chunks_by_id
    ]

    if not sources:
        # The model claimed a match but cited no chunk we actually
        # retrieved — cannot attribute the answer, so it isn't valid.
        return NoMatchResponse()

    return ChatAnswer(answer=result["answer"], sources=sources)


def answer_query(
    query: str, top_k: int = DEFAULT_TOP_K, persist_dir: str = PERSIST_DIR
) -> ChatAnswer | NoMatchResponse:
    """Full chat pipeline: retrieval -> branch -> (maybe) generation.

    Branches on retrieval.search()'s result BEFORE calling the generation
    LLM: an empty result short-circuits straight to NoMatchResponse, no
    LLM call made. A non-empty result still goes through generate_answer's
    second-layer grounding check, which may itself return NoMatchResponse.
    """
    chunks = search(query, top_k=top_k, persist_dir=persist_dir)
    if not chunks:
        return NoMatchResponse()
    return generate_answer(query, chunks)


class ChatRequest(BaseModel):
    question: str


class SourceModel(BaseModel):
    filename: str
    title: str | None = None


class ChatResponseModel(BaseModel):
    # "answer" -> answer + sources populated, message is None.
    # "no_match" -> message populated, answer/sources are None.
    type: str
    answer: str | None = None
    sources: list[SourceModel] | None = None
    message: str | None = None


def _to_response_model(result: ChatAnswer | NoMatchResponse) -> ChatResponseModel:
    if isinstance(result, ChatAnswer):
        return ChatResponseModel(
            type="answer",
            answer=result.answer,
            sources=[
                SourceModel(filename=s.filename, title=s.title) for s in result.sources
            ],
        )
    return ChatResponseModel(type="no_match", message=result.message)


def create_app() -> FastAPI:
    """Construct and return the web application.

    Exposes POST /chat for the frontend. CORS is wide open (allow_origins
    "*") since the frontend is plain static files with no build step —
    it may be opened directly via file:// or served from any static
    file server/port, and this is a local single-user tool, not a
    multi-tenant service, so there's no origin to lock down to.
    """
    app = FastAPI(title="ProofDesk API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat", response_model=ChatResponseModel)
    def chat(request: ChatRequest) -> ChatResponseModel:
        try:
            result = answer_query(request.question)
        except Exception:
            # Never leak internal exception details (stack traces, API
            # error bodies) to the client — log server-side, return a
            # generic message the frontend can show as-is.
            logger.exception("answer_query failed for question: %r", request.question)
            raise HTTPException(
                status_code=500,
                detail="Something went wrong answering that question. Please try again.",
            )
        return _to_response_model(result)

    return app


# Module-level instance so `uvicorn api.main:app` can find it directly.
app = create_app()
