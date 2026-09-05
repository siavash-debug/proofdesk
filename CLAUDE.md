# ProofDesk — RAG Assistant for Document/Certificate PDFs

## Purpose
A retrieval-augmented assistant for searching and answering questions over a
small personal set of document/certificate PDFs (certifications, credentials,
transcripts, etc.), used by both the owner and recruiters evaluating them.

## Architecture Decisions

### Scale: no heavy vector DB
The corpus is fewer than 20 documents. A managed vector database
(Pinecone, Weaviate, etc.) is unjustified operational overhead at this scale —
network latency, account setup, and cost outweigh any benefit. Use a
lightweight local solution instead: **ChromaDB in local/persistent mode**, or
a simple in-memory vector store if we want to avoid even that dependency.
Revisit only if the corpus grows by an order of magnitude.

**Embedding model in use:** ChromaDB's built-in `DefaultEmbeddingFunction`
(`chromadb.utils.embedding_functions.DefaultEmbeddingFunction`), which
resolves to `ONNXMiniLM_L6_V2` — this is **not** the `sentence-transformers`
Python package loaded directly. It is Chroma's own ONNX export of
`all-MiniLM-L6-v2`, downloaded from Chroma's S3 bucket on first use and run
via `onnxruntime` + `tokenizers`. Practically, this means:
- Same underlying model weights as `sentence-transformers/all-MiniLM-L6-v2`,
  but a different runtime — no PyTorch or `sentence-transformers` dependency
  at all, just `onnxruntime` + `tokenizers` (both installed transitively by
  `chromadb`).
- Smaller dependency footprint than loading `sentence-transformers`
  directly, at the cost of being tied to whatever ONNX export Chroma
  chooses to ship (version/behavior changes are Chroma's call, not ours,
  if we ever bump the `chromadb` version).
- No explicit embedding function is passed in `ingestion/embed.py` —
  Chroma applies this default automatically on `get_or_create_collection`.

### OCR pipeline for scanned PDFs
Many certificates are scans/images rather than text-layer PDFs, but not all —
some PDFs already have a usable text layer, and OCR would only introduce
noise for those. Pipeline (`ingestion/ocr.py`):
1. **Try the PDF's own text layer first**, via `pdfplumber`. A page counts as
   having a usable text layer if `pdfplumber` extracts at least
   `MIN_TEXT_LAYER_CHARS = 20` non-whitespace characters from it. If any page
   has one, that text is used directly and Tesseract is never invoked for
   that document (`ocr_used=False`).
2. **Only if no page has a usable text layer**, fall back to **Tesseract**:
   rasterize each page (`pdf2image`) and run OCR on the image
   (`ocr_used=True`).
3. Raw OCR output (when Tesseract was actually used) is noisy (line breaks
   mid-sentence, misread characters, lost structure). This text is then
   cleaned and restructured using the **gpt-oss-120b** model served via
   **Groq**, using `GROQ_API_KEY` from the environment.

**`ingestion/clean.py` splits into two independently-triggered
responsibilities**, not one combined "cleanup" step:
- **OCR error-correction** — only runs when `ocr_used=True`. Text pulled
  directly from a PDF's text layer has no OCR noise, so running correction
  instructions on it is pointless and risks the model "fixing" things that
  were never broken.
- **Structured field extraction** (`document_type`, `title`,
  `recipient_name`, `issue_date`, `certificate_number`,
  `issuing_organization`) — a semantic extraction task, independent of OCR
  status. This **always** runs via Groq, regardless of `ocr_used`, so every
  document gets structured metadata whether or not it needed OCR.

### Chunking Strategy
No generic fixed-size sliding-window chunker — documents here are short and
heterogeneous, and a fixed window would either split a certificate's facts
apart or fail to respect a resume's natural structure. Instead
(`ingestion/chunk.py`):
- **Certificates → one chunk per document.** These are short, and splitting
  one would separate facts that only make sense together (e.g.
  `recipient_name` from `certificate_number`).
- **Resumes → one chunk per section** (Summary, Experience, Education,
  Skills), split on the section headers already present in the text. Each
  section is an independently meaningful retrieval unit — a query about work
  history shouldn't also pull in unrelated skills text. A document is only
  treated as section-structured if at least 2 of the known headers are
  found as standalone lines (`MIN_SECTION_HEADERS_TO_SPLIT = 2`); otherwise
  it falls back to the single-chunk behavior above.

Every chunk carries forward all document-level metadata (filename,
chunk_id, document_type, title, recipient_name, issue_date,
certificate_number, issuing_organization) — this is what makes source
attribution possible at answer time.

#### Embedding Enrichment
Each chunk carries an `embedding_text` field, separate from the display
`text` field. **Why:** Phase 5 eval surfaced real false negatives (cases 4
and 5 in the golden dataset — Education and Skills questions) traced to
weak embedding representation for terse, list-style chunks: a bare bullet
list like the Skills section sits far from a natural-language question
("What technical skills does Ali have?") in embedding space, even though
it's the obviously correct answer.

**Fix:** for resume section chunks, `embedding_text` is prefixed with a
natural-language sentence describing the section before the original
content, e.g. *"This is Ali Sample's resume, Skills section — technical
skills section, listing his technical skills: [original bullet text]"*.
Only `embedding_text` is embedded (`ingestion/embed.py` computes the vector
from it explicitly, via `upsert(embeddings=...)`); the display `text` field
shown/cited to the user is completely unchanged. Certificate chunks (whole-
document, already prose-like) use `embedding_text == text` — no enrichment
needed there.

### No re-ranking (this phase)
With under 20 source documents, the retrieval candidate set is small enough
that a re-ranking stage adds complexity without a meaningful accuracy
benefit. Skip re-ranking entirely for now; revisit only if corpus size or
answer quality demands it later.

### Frontend: plain HTML (chat interface)
Proposal: a **single-page plain HTML/CSS/vanilla JS** chat UI rather than a
React app.

Reasoning:
- The corpus and feature set are both small and stable — one chat panel,
  one input box, message list, source citations. No routing, no complex
  client state, no component reuse pressure that would justify a framework.
- Zero build step (no bundler/toolchain to maintain) keeps the project easy
  to run and hand off — relevant since recruiters may be shown the source.
- Fewer dependencies means less to go wrong when demoing to a non-technical
  audience.
- If the UI later grows (multi-user auth, richer document browsing,
  streaming token-by-token rendering with complex state), migrating to React
  at that point is straightforward since the API layer is already decoupled
  (plain HTTP/JSON, or SSE for streaming).

If future requirements emerge (auth, multi-session state, more complex
interactions), reconsider React at that time — not preemptively.

**Implementation notes:** `api/main.py` exposes `POST /chat` (`{"question":
str}` → `{"type": "answer" | "no_match", "answer", "sources": [{filename,
title}], "message"}`) via FastAPI, with `CORSMiddleware(allow_origins=["*"])`
since the frontend is static files with no fixed origin (may be opened via
`file://` or served from any port). `api/main.py` calls `load_dotenv()` at
import time so `GROQ_API_KEY`/`GROQ_MODEL` are available when running the
server directly (`uvicorn api.main:app`), not just under pytest (which
loads `.env` itself in each test file). Unhandled exceptions in `/chat` are
caught and turned into a generic `HTTPException(500, ...)` with a safe
message — the real exception is logged server-side only, never sent to the
client. Source citations shown in the UI are filename-only (e.g. "Source:
cert_aws_sample.pdf") — no chunk text or chunk_id shown to the user, per
the "keep it simple" design call.

### How to run the full stack locally
1. **Start the API** (from the project root, with `.env` containing
   `GROQ_API_KEY`/`GROQ_MODEL`):
   ```
   python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
   ```
2. **Serve the frontend** as static files (needed for `fetch()` to work
   reliably — opening `index.html` directly via `file://` can behave
   inconsistently across browsers/sandboxes):
   ```
   cd frontend
   python -m http.server 5500
   ```
   Then open `http://127.0.0.1:5500/index.html` in a browser.
3. The frontend's `API_URL` in `frontend/app.js` is hardcoded to
   `http://127.0.0.1:8000/chat` — update it if the API runs on a
   different host/port.

### No-match handling: never guess when nothing is relevant
Because answers go to recruiters as well as the owner, a fabricated or
loosely-related answer is worse than no answer. The system must explicitly
detect "no sufficiently relevant source exists for this query" and respond
with a "not found" message, rather than letting the generation LLM
improvise from weak or irrelevant context.

This is enforced at two specific points, not as a vague guideline:

- **`retrieval/search.py` — the single enforcement point for relevance.**
  `search()` applies a `MIN_RELEVANCE_SCORE` threshold and drops any chunk
  that doesn't meet it. If no chunk in the corpus clears the bar for a
  given query, `search()` returns an empty list. This is the *only* place
  the threshold is evaluated — no other layer re-implements or
  second-guesses this decision.
- **`api/schemas.py` — an explicit, distinct response type for "no
  answer."** The API defines two separate response shapes: `ChatAnswer`
  (has `answer` + non-empty `sources`) and `NoMatchResponse` (fixed
  "not found" message, no answer/sources fields). The API branches on
  `search()`'s result *before* ever calling the generation LLM: an empty
  result short-circuits straight to `NoMatchResponse`. The generation LLM
  is never invoked when there's nothing relevant to ground it in — this
  is a hard requirement, not just a prompting convention (i.e. we do not
  rely on prompting the LLM to say "I don't know").

**`MIN_RELEVANCE_SCORE = 1.05`** (a Chroma distance — lower is more
similar, so this is a maximum-distance cutoff despite the name). Chosen by
running 6 calibration queries against the 3-document sample corpus (3
known-good, 3 known-bad):

| Query | Expected | Best-match distance |
|---|---|---|
| What AWS certification does Ali have? | match | 0.5527 |
| What Scrum certification did Ali complete? | match | 0.7523 |
| What is Ali's work experience with Kubernetes? | match | 1.0292 |
| Does Ali have a PhD in Physics? | no match | 1.0715 |
| What is Ali's favorite programming language? | no match | 1.3599 |
| What is the capital of France? | no match | 1.9873 |

1.05 sits just above the weakest legitimate match (1.0292) and rejects the
clearly unrelated query (1.9873). It deliberately does **not** try to
reject "Does Ali have a PhD in Physics?" (1.0715) at this layer — that
query sits essentially on top of the legitimate Kubernetes match, so no
single global distance threshold can cleanly separate them without also
risking real matches.

**This threshold intentionally favors recall over precision.** Retrieval's
only job is "don't discard anything that might be relevant" — filtering
out plausible-but-wrong matches (topically close, but the retrieved chunk
doesn't actually contain the answer) is handled by a second, independent
defense layer: the answer-generation prompt in `api/main.py` explicitly
instructs the LLM to verify the retrieved chunk actually *contains* the
answer, not merely that it's topically related, and to return the same
`NoMatchResponse` contract if it doesn't. Retrieval's threshold and
generation's grounding check are two separate lines of defense against a
fabricated/loosely-related answer — neither is relied on alone.

### Users and source attribution (mandatory)
The assistant is used by both the document owner and external recruiters.
Both audiences need trustworthy answers, so every answer must be
**attributable to a specific source document** (and ideally section).
This is a hard requirement, not a nice-to-have:
- Every chunk stored in the vector store must retain metadata: **source
  filename and section/heading** (if available). Page-level attribution is
  deliberately deprioritized — most source documents in this project are
  single-page, so a page number would add a field to track without adding
  meaningfully to attribution precision. This was a deliberate scope
  reduction made when implementing the chunk/citation schema, not an
  oversight: revisit if multi-page documents become common in the corpus.
- Every answer returned by the API must include citations referencing that
  metadata — never an unattributed answer.

### Phase 5: formal evaluation infrastructure (DeepEval)
Manual spot-checking (a handful of hand-picked queries) doesn't scale as
the pipeline gains more moving parts (chunking, retrieval threshold,
generation grounding). Phase 5 adds a repeatable, automated eval using
**DeepEval**, run via its pytest integration (`eval/run_eval.py`):

- **`eval/golden_dataset.py`** — a fixed list of test cases, each
  `{question, expected_behavior: "answer" | "no_match",
  expected_source_file (if answer), notes}`. Covers: 5 direct-fact
  questions (one per known fact across the 3 sample docs), the AWS ML
  Specialty borderline case (tests layer 2 — generation's grounding
  check), 2 irrelevant-query cases (test layer 1 — retrieval's
  `MIN_RELEVANCE_SCORE`), and 1 multi-fact synthesis case requiring
  chunks from more than one document (flagged in its `notes` as
  not-yet-validated, since multi-chunk synthesis was never explicitly
  built or tested).
- **Three metrics per case** (`eval/run_eval.py`):
  - **Correctness** — does the actual outcome (answer vs. no_match) match
    `expected_behavior`?
  - **Faithfulness** — for "answer" cases, is every claim in the
    generated answer actually grounded in the retrieved chunk (no
    hallucinated details)?
  - **Contextual Relevancy** — is the retrieved chunk actually relevant
    to the question (not just the answer correct despite noisy context)?
- **`requires_groq` skip marker** — every eval test is decorated with a
  `pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), ...)` marker
  (matching the same pattern used in `tests/test_ingestion.py`,
  `tests/test_api.py`, etc.), so the eval suite degrades to a skip rather
  than a hard failure in an environment without the key.

Eval runs consume real Groq API calls (generation + judge-model calls per
case) and are subject to Groq's daily token quota — a full 9-case run can
fail outright on `RateLimitError` if the day's quota is already consumed
by other work, which is a real operational constraint to plan around
when re-running this suite repeatedly in one day.

#### Latest eval round results (top_k=5, embedding_text enrichment active)
7/8 confirmed cases pass. 1 confirmed known limitation (cases #1 and #3 —
Contextual Relevancy failures on the AWS certification and Kubernetes
experience questions, root-caused to chunk granularity — see "Things to
monitor" below; this is accepted, not a regression). 1 case unconfirmed:

- **Case #2 (Scrum certification query) could not get a confirmed
  Contextual Relevancy score due to repeated Groq rate limiting during
  testing.** Correctness and Faithfulness both scored 1.00 on every
  attempt, and its retrieval distance (0.7523) is well within the
  calibrated range. Based on this partial evidence and the pattern
  matching case #5 (which did get a clean, passing Contextual Relevancy
  score), it is expected to pass Contextual Relevancy, but this is **not
  confirmed** with a clean run. Revisit if Groq quota/rate-limit
  conditions allow a clean isolated test.

Also confirmed by this round: the multi-fact synthesis case (#9),
flagged in `golden_dataset.py`'s notes as "may currently fail" since
multi-chunk synthesis had never been validated, actually **passes**
(Contextual Relevancy 0.85) — synthesis across chunks from multiple
documents works without any additional logic having been built for it.

Phase 5 is considered complete for this iteration as of this round.

## Things to monitor
- **`cert_aws_sample.pdf`'s en-dash character is unmapped in its embedded
  font, so `pdfplumber` extracts it as a replacement character (�).** This
  is baked into ingestion output and visible in the AWS cert answer text.
  Accepted as a known limitation — it's a cosmetic character-mapping
  issue that doesn't affect meaning or correctness, and fixing it would
  require either PDF-level font remapping or a text-cleanup regex pass,
  neither of which is worth the complexity at this scale. Not scheduled
  for fix.
- **OCR-correction consistency for repeated misread patterns** (e.g.
  multiple occurrences of the same OCR error in one document) showed
  ~50% failure rate before a prompt tightening, then 5/5 success after —
  but 5 runs is too small a sample to confirm the fix fully. Eval
  infrastructure now exists (`eval/golden_dataset.py`, `eval/run_eval.py`)
  — revisit by adding this as a dedicated eval case (run 20+ times) rather
  than an ad hoc `tests/test_ingestion.py` regression test.
- **`chromadb` version pin changed from a specific pinned version to
  `>=1.0`** in `requirements.txt`. The originally-pinned version had no
  prebuilt wheel for Python 3.14 (missing prebuilt `chroma-hnswlib` /
  `tokenizers` binaries for that interpreter version, and no Rust/MSVC
  toolchain available in this environment to build them from source), so
  the pin was relaxed to whatever resolves and installs cleanly on 3.14.
- **Contextual Relevancy metric remains low (~0.12-0.29) for cases 1 and
  3** (AWS certification question, Kubernetes experience question) in
  the Phase 5 golden dataset eval, due to chunk granularity — the
  Experience chunk mixes multiple jobs (Kubernetes/TechCorp and
  Terraform/CloudWorks), so part of any retrieved chunk is topically
  adjacent-but-irrelevant to a specific question. This is accepted as a
  known, low-severity limitation because Faithfulness and Correctness
  are both 1.00 for these cases — the answers themselves are accurate
  and fully grounded, this is purely a retrieval-context-precision
  metric being sensitive to intra-chunk noise. Splitting Experience into
  per-role chunks would fix this but is deprioritized in favor of
  higher-impact work.
- **`DEFAULT_TOP_K` in `retrieval/search.py` was tried at 3 (down from
  5)** to address the Contextual Relevancy issue above, on the theory
  that a 6-chunk corpus doesn't need top_k=5. Reverted: it didn't
  improve relevancy (the noise is inside individual chunks, not from
  extra low-relevance chunks pulled in by a higher top_k) and it broke
  the previously-passing multi-fact synthesis case (which needs 3+
  chunks from different documents to answer). `top_k=5` stays the
  default.

## Project Structure
```
/data        - input PDFs (sample/test data only, not real documents)
/ingestion   - OCR (Tesseract + Groq cleanup) + chunking + embedding
/retrieval   - vector search over the embedded chunks
/api         - backend service (chat endpoint, orchestrates retrieval + LLM)
/frontend    - plain HTML/CSS/JS chat interface
/tests       - tests for ingestion, retrieval, and API
/eval        - Phase 5 formal eval infra: golden dataset + DeepEval runner
```

## Environment Variables
- `GROQ_API_KEY` — required for both OCR text cleanup/extraction
  (`ingestion/clean.py`) and answer generation (`api/main.py`). The same
  key/provider is reused for both — no separate answer-generation provider
  was introduced; Groq's `gpt-oss-120b` handles both roles.
- `GROQ_MODEL` — the Groq model id to use (e.g. `openai/gpt-oss-120b`).
  Read via `os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")` in both
  `ingestion/clean.py` and `api/main.py`, so it falls back to
  `openai/gpt-oss-120b` if unset.

## Status
Ingestion (OCR + cleanup/extraction + chunking + embedding), retrieval, and
the API's two-layer generation logic are all implemented and covered by
tests (`tests/`) and a Phase 5 golden-dataset eval (`eval/`). Remaining
work: the frontend (still skeleton-only, per the "Frontend" architecture
decision above) and a final confirmatory full-9-case eval run at
`top_k=5` with `embedding_text` enrichment active — blocked on Groq's
daily token quota being exhausted from repeated eval reruns, not on any
known code issue.
