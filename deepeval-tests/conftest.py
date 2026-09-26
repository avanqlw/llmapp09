"""
Shared fixtures and configuration for deepeval LLM evaluation tests.
"""

import asyncio
import os
import re
import time

import pytest
import requests
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCaseParams


# ---------------------------------------------------------------------------
# Judge LLM: Gemini (free tier) with key + model fallback
# ---------------------------------------------------------------------------

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Flash-Lite has the highest free-tier throughput. Quotas are per model, so the
# second model gives a fresh quota when the first is exhausted.
DEFAULT_GEMINI_MODELS = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
RETRYABLE_STATUS = {429, 500, 503}


class GeminiJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by the Gemini REST API.

    Tries every (model, key) pair in order and moves on to the next one when
    a call is rate-limited (429) or the model is overloaded (5xx).
    """

    def __init__(self):
        self.api_keys = [
            key.strip()
            for key in (os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY_FALLBACK"))
            if key and key.strip()
        ]
        self.models = [
            m.strip()
            for m in os.getenv("GEMINI_MODELS", DEFAULT_GEMINI_MODELS).split(",")
            if m.strip()
        ]
        super().__init__(model_name=self.models[0])

    def load_model(self, *args, **kwargs):
        return self

    def get_model_name(self) -> str:
        return f"Gemini ({', '.join(self.models)})"

    def _call(self, prompt: str, json_mode: bool) -> str:
        if not self.api_keys:
            raise RuntimeError("Set GEMINI_API_KEY (and optionally GEMINI_API_KEY_FALLBACK).")
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        last_error = ""
        for attempt in range(3):
            for model in self.models:
                for key in self.api_keys:
                    resp = requests.post(
                        GEMINI_URL.format(model=model),
                        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                        json=body,
                        timeout=120,
                    )
                    if resp.status_code in RETRYABLE_STATUS:
                        last_error = f"{model}: HTTP {resp.status_code}"
                        continue
                    if resp.status_code in (401, 403) or "API_KEY_INVALID" in resp.text:
                        # Bad/expired key: fall through to the next key
                        last_error = f"{model}: invalid API key (HTTP {resp.status_code})"
                        continue
                    resp.raise_for_status()
                    parts = resp.json()["candidates"][0]["content"]["parts"]
                    return "".join(p.get("text", "") for p in parts)
            # Every model/key pair is rate-limited: wait for the per-minute window to reset
            time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"All Gemini models/keys rate-limited. Last error: {last_error}")

    def generate(self, prompt: str, schema=None):
        text = self._call(prompt, json_mode=schema is not None)
        if schema is None:
            return text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return schema.model_validate_json(match.group(0) if match else text)

    async def a_generate(self, prompt: str, schema=None):
        return await asyncio.to_thread(self.generate, prompt, schema)


JUDGE = GeminiJudge()


# ---------------------------------------------------------------------------
# Reusable GEval metric factories
# ---------------------------------------------------------------------------

def json_schema_metric(schema_description: str):
    """Creates a GEval metric that checks JSON schema compliance."""
    return GEval(
        model=JUDGE,
        name="JSON Schema Compliance",
        criteria=(
            "Evaluate whether the actual output is valid JSON that conforms to "
            "the required schema. Only check structure, key names, and data "
            "types — do NOT penalize for specific values. "
            + schema_description
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


def output_correctness_metric():
    """Creates a GEval metric that checks factual/logical correctness."""
    return GEval(
        model=JUDGE,
        name="Output Correctness",
        criteria=(
            "Determine whether the actual output is logically correct and "
            "reasonable given the input text. The analysis should make sense "
            "for the provided input."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )


def answer_relevancy_metric():
    """Creates a GEval metric that checks whether the output is topically
    relevant to the input.  Unlike AnswerRelevancyMetric (which assumes a
    Q&A format), this works for classification and analysis endpoints where
    the output is structured metadata about the input text."""
    return GEval(
        model=JUDGE,
        name="Answer Relevancy",
        criteria=(
            "Evaluate whether the actual output is topically relevant to the "
            "input text. The labels, categories, or analysis in the output "
            "should directly relate to the subject matter of the input. "
            "Structured metadata (labels, categories, confidence scores) that "
            "accurately describes the input text should be considered relevant."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
    )
