# ProofDesk

A retrieval-augmented (RAG) assistant for searching and answering questions
over a small set of credential/certificate PDFs — built to demonstrate
production-grade RAG practices at a scale where they're usually skipped:
OCR fallback for scanned documents, a two-layer defense against
hallucinated answers, and a formal, automated evaluation harness instead
of ad hoc spot-checking. Answers are always grounded in a retrieved
document and cited by filename, or the assistant explicitly says it
doesn't know — it never guesses.

## Architecture

```
PDF ──▶ OCR ──▶ Clean/Extract ──▶ Chunk ──▶ Embed ──▶ Vector Store (Chroma)
       (pdfplumber                (structured             │
        first, Tesseract           fields via                │
        fallback)                  Groq LLM)                 ▼
                                                          Retrieval
                                                     (similarity search +
                                                      relevance threshold)
                                                              │
                                                              ▼
                                                        FastAPI backend
                                                     (POST /chat — grounding
                                                      verification before
                                                      any answer is generated)
                                                              │
                                                              ▼
                                                   Plain HTML/CSS/JS frontend
                                                     (chat UI, source citations)
```

## Key engineering decisions

- **Two-layer defense against hallucination.** Retrieval uses a
  deliberately loose relevance threshold tuned to favor recall (it would
  rather return a borderline match than miss a real one). A second,
  independent layer — the generation prompt itself — is explicitly
  instructed to verify the retrieved chunk actually *contains* the
  answer, not just that it's topically related, and to say "not found"
  otherwise. Neither layer is trusted alone.
- **LLM-based OCR error-correction, iteratively tuned against a known
  failure case.** Tesseract OCR output is cleaned up by an LLM (Groq's
  gpt-oss-120b) rather than regex heuristics, since misreads like a
  Roman numeral "I" rendered as a stray "\|" need contextual judgment to
  fix. The correction prompt was refined against a real recurring
  failure pattern (fixing the first occurrence of a misread but missing
  repeats of it later in the same document) until it held up reliably.
- **Embedding enrichment fix, found via evaluation, not guesswork.**
  Formal eval surfaced real false negatives on terse, list-style content
  (a bare "Skills" bullet list embeds poorly against a natural-language
  question). Fixed by embedding a distinct, LLM-friendly description of
  the section separately from the text actually shown to the user —
  the display text is never altered, only what gets embedded.
- **Formal, automated evaluation with a golden dataset**, not manual
  spot-checks. A fixed set of test cases — direct-fact questions,
  borderline "plausible but wrong" queries, fully irrelevant queries,
  and a multi-document synthesis case — is run through DeepEval on
  every meaningful pipeline change, scored on Correctness, Faithfulness,
  and Contextual Relevancy, so regressions are caught by a rerun instead
  of a hunch.
- **Chunking strategy matched to document shape, not generic
  fixed-size windows.** Short certificates are kept as one chunk (their
  facts only make sense together); resumes are split by section, since
  a "Skills" question shouldn't pull in unrelated work history.

## How to run

**1. Start the API** (from the project root, with `.env` containing
`GROQ_API_KEY`/`GROQ_MODEL`):
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**2. Serve the frontend as static files** (don't open `index.html`
directly via `file://` — serve it, or `fetch()` can behave
inconsistently):
```bash
cd frontend
python -m http.server 5500
```
Then open `http://127.0.0.1:5500/index.html` in a browser.

**3. Re-run ingestion** (OCR → clean → chunk → embed) or the test/eval
suites — see [`CLAUDE.md`](CLAUDE.md) for the full command reference and
environment variable list.

## A note on the data

This project uses **sample, synthetic documents** (a fictional resume and
two fictional certificates) — never real personal or credential data.
That was a deliberate privacy-by-design decision made early on: an
assistant whose entire purpose is handling sensitive personal documents
shouldn't be developed or demoed against real ones, especially given
this repo's own eval logs, embeddings, and test fixtures need to be
freely runnable and shareable without any data-handling risk.

## Full architecture decision history

Every design decision above — and the ones that didn't make this
summary, including known limitations and things intentionally left
unfixed — is documented in depth in [`CLAUDE.md`](CLAUDE.md), written
as the project was built rather than reconstructed after the fact.
