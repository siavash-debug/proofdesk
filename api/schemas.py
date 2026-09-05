"""Request/response data shapes for the chat API.

A ChatResponse must always carry a non-empty list of source citations —
answers without attribution are not valid responses in this system.

No-match contract (see CLAUDE.md "No-match handling"):
the API has exactly two possible response shapes for a chat query, and
they are structurally distinct types, not a flag on one type:

- ChatAnswer: retrieval.search() returned at least one chunk meeting
  MIN_RELEVANCE_SCORE, AND the generation LLM confirmed that chunk
  actually contains the answer (not just topically related). Carries
  `answer: str` and `sources: list[Citation]` (non-empty).
- NoMatchResponse: either retrieval.search() returned an empty list, OR
  the generation LLM determined none of the retrieved chunks actually
  answer the question. Carries a fixed "not found" message and no
  `answer`/`sources` fields.

The API layer must branch on retrieval results BEFORE any call to the
generation LLM. It is not permitted to call the LLM and then decide
after the fact whether the answer was groundless — the absence of
sufficiently relevant chunks is decided entirely in retrieval.search()
(via MIN_RELEVANCE_SCORE), and the API's only job is to act on that
result. The generation LLM's grounding check is a *second*, independent
layer on top of that — it can still produce NoMatchResponse even when
search() returned candidates, if none of them actually answer the
question.
"""

from dataclasses import dataclass

NOT_FOUND_MESSAGE = (
    "I couldn't find information about that in the available documents."
)


@dataclass
class Citation:
    filename: str
    chunk_id: str
    title: str | None = None


@dataclass
class ChatAnswer:
    answer: str
    sources: list[Citation]


@dataclass
class NoMatchResponse:
    message: str = NOT_FOUND_MESSAGE
