"""Golden evaluation dataset for the RAG pipeline (Phase 5 eval infra).

Each case:
    {
        "question": str,
        "expected_behavior": "answer" | "no_match",
        "expected_source_file": str | None,  # required if "answer"
        "notes": str,
    }

Covers, per CLAUDE.md's two-layer no-match defense:
- 5 direct-fact questions (one per known fact across the 3 sample docs)
- The AWS ML Specialty borderline case (tests layer 2: generation's
  grounding check, since retrieval alone lets this one through)
- 2 irrelevant-query cases (test layer 1: retrieval's MIN_RELEVANCE_SCORE)
- 1 multi-fact synthesis case (flagged as may-currently-fail — no
  multi-chunk synthesis logic has been built/verified yet)
"""

GOLDEN_DATASET = [
    {
        "question": "What AWS certification does Ali have?",
        "expected_behavior": "answer",
        "expected_source_file": "cert_aws_sample.pdf",
        "notes": "Direct fact: AWS Certified Solutions Architect - Associate.",
    },
    {
        "question": "What Scrum certification did Ali complete?",
        "expected_behavior": "answer",
        "expected_source_file": "cert_scanned_sample.pdf",
        "notes": (
            "Direct fact: Professional Scrum Master I (PSM I). Also "
            "exercises the OCR + cleanup path, not just text-layer PDFs."
        ),
    },
    {
        "question": "What is Ali's work experience with Kubernetes?",
        "expected_behavior": "answer",
        "expected_source_file": "resume_ali_sample.pdf",
        "notes": (
            "Direct fact from the Experience section chunk (CI/CD "
            "migration to Kubernetes, 60% deployment time reduction)."
        ),
    },
    {
        "question": "What did Ali study and where?",
        "expected_behavior": "answer",
        "expected_source_file": "resume_ali_sample.pdf",
        "notes": (
            "Direct fact from the Education section chunk (B.Sc. Computer "
            "Engineering, Sharif University of Technology)."
        ),
    },
    {
        "question": "What technical skills does Ali have?",
        "expected_behavior": "answer",
        "expected_source_file": "resume_ali_sample.pdf",
        "notes": "Direct fact from the Skills section chunk.",
    },
    {
        "question": "Does Ali have an AWS Machine Learning Specialty certification?",
        "expected_behavior": "no_match",
        "expected_source_file": None,
        "notes": (
            "LAYER 2 TEST. Retrieval returns the AWS cert chunk with a "
            "strong match (distance ~0.63, well under MIN_RELEVANCE_SCORE "
            "of 1.05) because it's the closest chunk topically, but that "
            "chunk is 'Solutions Architect - Associate', not 'Machine "
            "Learning Specialty'. Only the generation-layer grounding "
            "check (api/main.py) can catch this; retrieval's threshold "
            "cannot, since the retrieved chunk is a legitimate top match."
        ),
    },
    {
        "question": "What is the capital of France?",
        "expected_behavior": "no_match",
        "expected_source_file": None,
        "notes": (
            "LAYER 1 TEST. Totally unrelated to the corpus. Calibration "
            "distance was 1.9873, far above MIN_RELEVANCE_SCORE (1.05) - "
            "retrieval alone should reject this with an empty result."
        ),
    },
    {
        "question": "Does Ali have a PhD in Physics?",
        "expected_behavior": "no_match",
        "expected_source_file": None,
        "notes": (
            "LAYER 1 TEST. Plausible-sounding but absent from all 3 docs. "
            "Calibration distance was 1.0715, just above MIN_RELEVANCE_SCORE "
            "(1.05) - retrieval alone should reject this with an empty "
            "result (see CLAUDE.md calibration table)."
        ),
    },
    {
        "question": (
            "What certifications does Ali have and what technical skills "
            "does he bring to a DevOps role?"
        ),
        "expected_behavior": "answer",
        "expected_source_file": None,  # spans multiple documents/chunks
        "notes": (
            "MULTI-FACT SYNTHESIS CASE - FLAGGED AS MAY CURRENTLY FAIL. "
            "Requires combining information from at least two chunks "
            "(both certificate documents for 'certifications', plus the "
            "resume's Skills chunk for 'technical skills'). We have not "
            "built or verified multi-chunk synthesis logic - "
            "retrieval.search() returns up to top_k=5 chunks and all are "
            "passed to generation, so synthesis MAY already work "
            "incidentally, but this has not been validated and is exactly "
            "the kind of gap this eval is meant to surface, not paper over."
        ),
    },
]
