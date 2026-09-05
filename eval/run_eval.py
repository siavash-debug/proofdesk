"""Phase 5 golden-dataset evaluation, using DeepEval's pytest integration.

Runs every case in eval/golden_dataset.py through the real pipeline
(retrieval.search -> api.main.answer_query) against a throwaway Chroma
collection built from the 3 sample docs (never the real /chroma_db), and
scores each "answer" case on three axes:

- Correctness: did the actual outcome (answer vs no_match, and — for
  answer cases — the right source file) match what was expected? This is
  a deterministic check, not an LLM judgment (implemented as a small
  custom DeepEval metric, CorrectnessMetric, below).
- Faithfulness: for "answer" cases only, is every claim in the generated
  answer actually grounded in the retrieved chunk (DeepEval's built-in
  FaithfulnessMetric)?
- Contextual Relevancy: for "answer" cases only, is the retrieved chunk
  actually relevant to the question (DeepEval's built-in
  ContextualRelevancyMetric)?

For "no_match" cases, only Correctness applies — Faithfulness/Contextual
Relevancy require an actual_output and retrieval_context to judge, which
a NoMatchResponse doesn't produce.

Uses eval/groq_model.py (Groq's gpt-oss-120b) as the judge model for the
built-in metrics, so no separate OPENAI_API_KEY is needed.

Run with:
    pytest eval/run_eval.py -v -s
or directly:
    python eval/run_eval.py
"""

import os
import shutil

import pytest
from dotenv import load_dotenv

from deepeval.metrics import BaseMetric, ContextualRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from api.main import answer_query
from api.schemas import ChatAnswer, NoMatchResponse
from eval.golden_dataset import GOLDEN_DATASET
from eval.groq_model import GroqEvalModel
from ingestion.chunk import chunk_document
from ingestion.clean import clean_ocr_text
from ingestion.embed import embed_and_store
from ingestion.ocr import extract_raw_text
from retrieval.search import search

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_PDFS = ["cert_aws_sample.pdf", "resume_ali_sample.pdf", "cert_scanned_sample.pdf"]

requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set in environment/.env",
)

# Collected across all parametrized test runs so a full report can be
# printed at the end of the session (see pytest_sessionfinish below).
RESULTS: list[dict] = []


class CorrectnessMetric(BaseMetric):
    """Deterministic check: did the pipeline's outcome match expectations?

    Not an LLM-judged metric — "answer vs no_match" and "right source
    file" are facts we can check directly against what the pipeline
    actually returned, so there's no reason to spend an LLM call judging
    something we already know exactly.
    """

    def __init__(self, expected_behavior: str, expected_source_file: str | None):
        self.threshold = 1.0
        self.expected_behavior = expected_behavior
        self.expected_source_file = expected_source_file
        self.actual_behavior = None
        self.actual_source_files: list[str] = []

    @property
    def __name__(self):
        return "Correctness"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        behavior_ok = self.actual_behavior == self.expected_behavior
        source_ok = (
            self.expected_source_file is None
            or self.expected_source_file in self.actual_source_files
        )
        self.score = 1.0 if (behavior_ok and source_ok) else 0.0
        reasons = []
        if not behavior_ok:
            reasons.append(
                f"expected behavior '{self.expected_behavior}', got "
                f"'{self.actual_behavior}'"
            )
        if not source_ok:
            reasons.append(
                f"expected source '{self.expected_source_file}' not among "
                f"cited sources {self.actual_source_files}"
            )
        self.reason = "; ".join(reasons) if reasons else "outcome matched expectations"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return bool(self.success)


@pytest.fixture(scope="session")
def eval_collection_dir(tmp_path_factory):
    """Embed all 3 sample docs into one throwaway Chroma dir for the whole run."""
    all_chunks = []
    for filename in SAMPLE_PDFS:
        ocr_result = extract_raw_text(os.path.join(DATA_DIR, filename))
        cleaned = clean_ocr_text(ocr_result["text"], ocr_result["ocr_used"])
        document = {"filename": filename, **cleaned}
        all_chunks.extend(chunk_document(document))

    persist_dir = str(tmp_path_factory.mktemp("eval_chroma_db"))
    embed_and_store(all_chunks, persist_dir=persist_dir)
    yield persist_dir
    shutil.rmtree(persist_dir, ignore_errors=True)


@requires_groq
@pytest.mark.parametrize(
    "case", GOLDEN_DATASET, ids=[c["question"] for c in GOLDEN_DATASET]
)
def test_golden_case(case, eval_collection_dir):
    question = case["question"]
    expected_behavior = case["expected_behavior"]
    expected_source_file = case["expected_source_file"]

    retrieved_chunks = search(question, persist_dir=eval_collection_dir)
    response = answer_query(question, persist_dir=eval_collection_dir)

    actual_behavior = "answer" if isinstance(response, ChatAnswer) else "no_match"
    actual_source_files = (
        [s.filename for s in response.sources] if isinstance(response, ChatAnswer) else []
    )

    correctness = CorrectnessMetric(expected_behavior, expected_source_file)
    correctness.actual_behavior = actual_behavior
    correctness.actual_source_files = actual_source_files
    correctness.measure(None)

    metric_results = {"Correctness": correctness}

    if isinstance(response, ChatAnswer):
        retrieval_context = [c["text"] for c in retrieved_chunks] or [
            s.chunk_id for s in response.sources
        ]
        test_case = LLMTestCase(
            input=question,
            actual_output=response.answer,
            retrieval_context=retrieval_context,
        )

        judge_model = GroqEvalModel()
        faithfulness = FaithfulnessMetric(
            threshold=0.5, model=judge_model, async_mode=False
        )
        contextual_relevancy = ContextualRelevancyMetric(
            threshold=0.5, model=judge_model, async_mode=False
        )

        for name, metric in [
            ("Faithfulness", faithfulness),
            ("Contextual Relevancy", contextual_relevancy),
        ]:
            try:
                metric.measure(test_case)
            except Exception as exc:  # judge-model call failed; record, don't crash the run
                metric.score = 0.0
                metric.success = False
                metric.reason = f"metric errored: {exc}"
            metric_results[name] = metric

    overall_pass = all(m.is_successful() for m in metric_results.values())

    RESULTS.append(
        {
            "question": question,
            "expected_behavior": expected_behavior,
            "actual_behavior": actual_behavior,
            "overall_pass": overall_pass,
            "metrics": {
                name: {
                    "score": m.score,
                    "success": m.is_successful(),
                    "reason": m.reason,
                }
                for name, m in metric_results.items()
            },
            "notes": case["notes"],
        }
    )

    failed_metrics = [name for name, m in metric_results.items() if not m.is_successful()]
    assert not failed_metrics, (
        f"case failed on: {', '.join(failed_metrics)}\n"
        + "\n".join(f"  {name}: {metric_results[name].reason}" for name in failed_metrics)
    )


def _safe_print(text: str) -> None:
    # DeepEval's judge-model reasons sometimes contain Unicode punctuation
    # (e.g. non-breaking hyphens) that the Windows console's cp1252
    # encoding can't represent, which crashes print() outright when
    # stdout is redirected to a file. Fall back to a lossy but
    # non-crashing encode/decode round-trip in that case.
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(__import__("sys").stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding))


@pytest.fixture(scope="session", autouse=True)
def _print_report_at_session_end():
    # A plain pytest_sessionfinish hook is only picked up from conftest.py
    # or a plugin, not from an ordinary test module — this autouse
    # session-scoped fixture's teardown is the reliable equivalent here.
    yield
    if not RESULTS:
        return
    _safe_print("\n\n" + "=" * 100)
    _safe_print("GOLDEN DATASET EVAL REPORT")
    _safe_print("=" * 100)
    for result in RESULTS:
        status = "PASS" if result["overall_pass"] else "FAIL"
        _safe_print(f"\n[{status}] {result['question']}")
        _safe_print(
            f"  expected={result['expected_behavior']}  actual={result['actual_behavior']}"
        )
        for metric_name, info in result["metrics"].items():
            mark = "ok" if info["success"] else "FAILED"
            score_str = f"{info['score']:.2f}" if info["score"] is not None else "n/a"
            _safe_print(f"    - {metric_name}: {mark} (score={score_str}) {info['reason']}")
    passed = sum(1 for r in RESULTS if r["overall_pass"])
    _safe_print(f"\n{'=' * 100}\nTOTAL: {passed}/{len(RESULTS)} cases passed\n{'=' * 100}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
