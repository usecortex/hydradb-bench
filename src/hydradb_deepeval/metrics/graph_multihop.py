from __future__ import annotations

import json
import re
from typing import Optional, Union

from deepeval.metrics import BaseMetric
from deepeval.models import DeepEvalBaseLLM, GPTModel
from deepeval.test_case import LLMTestCase


class GraphMultiHopAccuracyMetric(BaseMetric):
    """LLM-as-a-judge metric that verifies the retrieved context contains the
    intermediate reasoning steps required for multi-hop graph traversal."""

    def __init__(
        self,
        threshold: float = 0.8,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
    ):
        self.threshold = threshold
        self.include_reason = include_reason
        self.score = 0.0
        self.reason = ""
        self.success = False

        # Resolve model: wrap plain strings (e.g. "gpt-4o") in GPTModel
        if isinstance(model, str):
            self.model = GPTModel(model=model)
        elif model is None:
            self.model = GPTModel()
        else:
            self.model = model

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # 1. Validate prerequisites
        if not test_case.retrieval_context:
            self.success = False
            self.score = 0.0
            self.reason = "Missing retrieval context."
            return self.score

        intermediate_steps = (
            test_case.additional_metadata.get("intermediate_steps", [])
            if test_case.additional_metadata
            else []
        )
        if not intermediate_steps:
            self.success = False
            self.score = 0.0
            self.reason = "Missing intermediate steps in test case metadata."
            return self.score

        # 2. Construct the LLM-as-a-Judge prompt
        prompt = f"""You are an expert evaluator for Graph RAG systems.
Your task is to evaluate if the retrieved context contains the necessary intermediate reasoning steps to answer the question.

Question: {test_case.input}
Required Intermediate Steps: {json.dumps(intermediate_steps)}
Retrieved Context: {json.dumps(test_case.retrieval_context)}

Evaluate if the Retrieved Context sufficiently covers the Required Intermediate Steps.
Output ONLY a valid JSON object with the following keys:
- "score": A float between 0.0 and 1.0 (1.0 if all steps are clearly supported, 0.0 if none are).
- "reason": A concise explanation of your score, citing which steps were found or missing.

JSON Output:
"""

        # 3. Call the LLM Judge (Async)
        try:
            result = await self.model.a_generate(prompt)
            # DeepEval's a_generate returns a (response_text, cost) tuple;
            # some custom models return a bare string. Handle both.
            res = result[0] if isinstance(result, tuple) else result
        except Exception as e:
            self.success = False
            self.score = 0.0
            self.reason = f"LLM generation failed: {e}"
            return self.score

        # 4. Parse the response with robust error handling
        try:
            clean_res = re.sub(r"^```json\s*|\s*```$", "", res.strip(), flags=re.MULTILINE)
            result_json = json.loads(clean_res)

            self.score = float(result_json.get("score", 0.0))
            self.reason = result_json.get("reason", "No reason provided by the judge.")
            self.success = self.score >= self.threshold

        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            self.success = False
            self.score = 0.0
            self.reason = (
                f"Failed to parse LLM judge response as JSON: {e} | "
                f"Raw response: {str(res)[:200]}"
            )

        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_measure(test_case))
        # Fallback when called from inside a running event loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.a_measure(test_case)).result()

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:
        return "Graph Multi-Hop Accuracy"