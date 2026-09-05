"""Clean OCR errors and extract structured fields via gpt-oss-120b on Groq.

Reads GROQ_API_KEY from the environment. Two distinct responsibilities,
kept separate because they have different triggers:

1. OCR error-correction — only relevant when the text came from Tesseract
   (ocr_used=True). Text pulled from a PDF's own text layer has no OCR
   noise, so running correction instructions on it is pointless and
   risks the model "fixing" things that were never broken.
2. Structured field extraction (document_type, title, recipient_name,
   issue_date, certificate_number, issuing_organization) — a semantic
   extraction task that applies to any document text, OCR'd or not.
   This always runs, regardless of ocr_used.
"""

import json
import os

from groq import Groq

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

METADATA_FIELDS = [
    "document_type",
    "title",
    "recipient_name",
    "issue_date",
    "certificate_number",
    "issuing_organization",
]

_FIELD_DESCRIPTIONS = """- document_type: e.g. "certificate", "resume", "transcript"
- title: the document's title or the name of the credential/award
- recipient_name: the person the document is issued to or about
- issue_date: the date the document was issued, in the format found in \
the text (do not invent a date if none is present; use null)
- certificate_number: certificate/credential ID if present, else null
- issuing_organization: the organization that issued the document"""

# Used when ocr_used=True: fixes OCR misreads AND extracts fields.
SYSTEM_PROMPT_CORRECT_AND_EXTRACT = f"""You are cleaning up raw OCR text extracted from a scanned \
certificate or document. The text may contain OCR errors: misread \
characters, broken words, lost line structure, or stray noise.

Fix OCR errors using context, then extract the following fields as \
JSON, with exactly these keys plus "cleaned_full_text":

A common Tesseract misread on these certificates: a Roman numeral "I" \
(as in credential-level suffixes like "Master I", "Level I", "Part I") is \
often misread as a stray pipe character "|" or the digit "1" immediately \
after the word. For example, "Master |" or "Master 1" at the end of a \
title almost always means "Master I" — restore the "I", do not simply \
delete the stray character.

Apply this correction to EVERY occurrence of the pattern in the text, \
not just the first one you encounter — scan the full text for all \
instances before finalizing your output.

{_FIELD_DESCRIPTIONS}
- cleaned_full_text: the full document text, with OCR errors corrected \
and structure restored, but no facts added or removed

If a field cannot be determined from the text, use null for that field \
(cleaned_full_text should never be null). Respond with JSON only."""

# Used when ocr_used=False: extraction only, no correction instructions —
# the text is already clean, so there's nothing to fix and no
# cleaned_full_text to produce (the caller keeps the original text as-is).
SYSTEM_PROMPT_EXTRACT_ONLY = f"""You are extracting structured fields from a document's text. \
The text is already clean (not OCR output) — do not rewrite, correct, or \
alter it in any way, only extract information from it.

Extract the following fields as JSON, with exactly these keys:

{_FIELD_DESCRIPTIONS}

If a field cannot be determined from the text, use null for that field. \
Respond with JSON only."""


def _call_groq(system_prompt: str, text: str) -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(response.choices[0].message.content)


def clean_ocr_text(raw_text: str, ocr_used: bool) -> dict:
    """Extract structured metadata, correcting OCR errors when needed.

    Always calls the Groq model to extract METADATA_FIELDS, since field
    extraction is a semantic task independent of OCR status.

    If ocr_used is True, the same call also corrects OCR misreads and
    returns the corrected text as cleaned_full_text.

    If ocr_used is False, no correction is requested (the text is already
    clean) and cleaned_full_text is set to raw_text unchanged — avoiding
    any risk of the model altering already-correct text.
    """
    if ocr_used:
        result = _call_groq(SYSTEM_PROMPT_CORRECT_AND_EXTRACT, raw_text)
        cleaned_full_text = result.get("cleaned_full_text") or raw_text
    else:
        result = _call_groq(SYSTEM_PROMPT_EXTRACT_ONLY, raw_text)
        cleaned_full_text = raw_text

    return {field: result.get(field) for field in METADATA_FIELDS} | {
        "cleaned_full_text": cleaned_full_text
    }
