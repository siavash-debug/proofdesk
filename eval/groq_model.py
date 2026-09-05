"""A DeepEval-compatible LLM wrapper around Groq's gpt-oss-120b.

DeepEval's built-in metrics (Faithfulness, Contextual Relevancy, etc.)
default to calling OpenAI, which would require a separate OPENAI_API_KEY
we don't otherwise need in this project. Since Groq's client is
OpenAI-API-compatible, this wraps it as a DeepEvalBaseLLM so eval metrics
reuse the same GROQ_API_KEY / GROQ_MODEL already configured for
ingestion and generation (see ingestion/clean.py, api/main.py).
"""

import json
import os

from deepeval.models.base_model import DeepEvalBaseLLM
from groq import Groq


class GroqEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.environ.get(
            "GROQ_MODEL", "openai/gpt-oss-120b"
        )
        super().__init__(model=self.model_name)

    def load_model(self):
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    def get_model_name(self) -> str:
        return self.model_name

    def _call(self, prompt: str, schema=None) -> str:
        client = self.model
        kwargs = {}
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return completion.choices[0].message.content

    def generate(self, prompt: str, schema=None):
        output = self._call(prompt, schema=schema)
        if schema is not None:
            return schema.model_validate(json.loads(output))
        return output

    async def a_generate(self, prompt: str, schema=None):
        # No async Groq client wired up; DeepEval metrics fall back to
        # the sync path when async_mode=False is set on each metric.
        return self.generate(prompt, schema=schema)
