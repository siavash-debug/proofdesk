"""Split a cleaned document into retrieval chunks.

Chunking strategy (no generic fixed-size sliding window — documents here
are short and heterogeneous):

- If the document's text contains at least two of the known resume
  section headers (Summary, Experience, Education, Skills) as standalone
  lines, split it into one chunk per section. Each section is an
  independently meaningful retrieval unit (e.g. a query about work
  history shouldn't also pull in unrelated skills text).
- Otherwise (certificates and anything else), the whole document is ONE
  chunk. These documents are short, and splitting a certificate would
  separate facts that only make sense together (e.g. recipient_name from
  certificate_number).

Every chunk carries forward all document-level metadata (filename,
chunk_id, document_type, title, recipient_name, issue_date,
certificate_number, issuing_organization) plus its own `text` — this is
what makes source attribution possible at answer time.

Each chunk also carries an `embedding_text` field, separate from `text`.
Short, terse, list-style chunks (e.g. a bare "Skills" bullet list) sit far
from natural-language questions in embedding space — a real false-
negative cause found during Phase 5 eval (see CLAUDE.md). `embedding_text`
prefixes such chunks with a natural-language sentence describing what the
section is, which is what actually gets embedded (see ingestion/embed.py)
— `text` stays exactly as-is for what's shown/cited to the user.
"""

import re

# Section headers recognized in resume-style documents, matched as a
# standalone line (case-insensitive, optional trailing colon/whitespace).
SECTION_HEADERS = ["Summary", "Experience", "Education", "Skills"]
_SECTION_HEADER_SET = {h.lower() for h in SECTION_HEADERS}

# Natural-language description of what each resume section contains, used
# to build embedding_text — phrased so the embedding sits closer to how a
# person would actually ask about that section.
_SECTION_DESCRIPTIONS = {
    "summary": "professional summary, giving an overview of his background",
    "experience": "work experience section, detailing his professional history and roles",
    "education": "education section, listing his academic background and degrees",
    "skills": "technical skills section, listing his technical skills",
}

# A document is treated as section-structured only if at least this many
# of the known headers are actually present — a single incidental match
# (e.g. a certificate that happens to contain the word "Skills") isn't
# enough to justify splitting it.
MIN_SECTION_HEADERS_TO_SPLIT = 2


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "chunk"


def _find_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (header, section_text) pairs on known section headers.

    Returns [] if fewer than MIN_SECTION_HEADERS_TO_SPLIT headers are found
    as standalone lines.
    """
    lines = text.splitlines()
    header_line_indices = [
        i
        for i, line in enumerate(lines)
        if line.strip().rstrip(":").lower() in _SECTION_HEADER_SET
    ]

    if len(header_line_indices) < MIN_SECTION_HEADERS_TO_SPLIT:
        return []

    sections = []
    for idx, line_no in enumerate(header_line_indices):
        header = lines[line_no].strip().rstrip(":")
        start = line_no + 1
        end = (
            header_line_indices[idx + 1]
            if idx + 1 < len(header_line_indices)
            else len(lines)
        )
        section_text = "\n".join(lines[start:end]).strip()
        sections.append((header, section_text))

    return sections


def chunk_document(document: dict) -> list[dict]:
    """Split a cleaned document into chunks.

    `document` is the merged ocr + clean.py output for one file:
        {
            "filename": str,
            "document_type": str | None,
            "title": str | None,
            "recipient_name": str | None,
            "issue_date": str | None,
            "certificate_number": str | None,
            "issuing_organization": str | None,
            "cleaned_full_text": str,
        }

    Returns a list of chunk dicts, each of the form:
        {
            "filename": str,
            "chunk_id": str,
            "document_type": str | None,
            "title": str | None,
            "recipient_name": str | None,
            "issue_date": str | None,
            "certificate_number": str | None,
            "issuing_organization": str | None,
            "text": str,            # display/citation text, unchanged
            "embedding_text": str,  # what actually gets embedded
        }
    """
    filename = document["filename"]
    text = document["cleaned_full_text"]
    doc_slug = _slugify(filename.rsplit(".", 1)[0])

    base_metadata = {
        "filename": filename,
        "document_type": document.get("document_type"),
        "title": document.get("title"),
        "recipient_name": document.get("recipient_name"),
        "issue_date": document.get("issue_date"),
        "certificate_number": document.get("certificate_number"),
        "issuing_organization": document.get("issuing_organization"),
    }

    sections = _find_sections(text)

    if not sections:
        full_text = text.strip()
        return [
            {
                **base_metadata,
                "chunk_id": f"{doc_slug}::full",
                "text": full_text,
                "embedding_text": full_text,
            }
        ]

    recipient_name = document.get("recipient_name") or "the candidate"
    chunks = []
    for header, section_text in sections:
        chunk_text = f"{header}\n{section_text}".strip()
        description = _SECTION_DESCRIPTIONS.get(
            header.lower(), f"{header.lower()} section"
        )
        embedding_text = (
            f"This is {recipient_name}'s resume, {header} section — "
            f"{description}: {section_text}".strip()
        )
        chunks.append(
            {
                **base_metadata,
                "chunk_id": f"{doc_slug}::{_slugify(header)}",
                "text": chunk_text,
                "embedding_text": embedding_text,
            }
        )
    return chunks
