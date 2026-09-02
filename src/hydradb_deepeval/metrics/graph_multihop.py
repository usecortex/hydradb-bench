import json
import re
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM

class GraphMultiHopAccuracyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.8,
        model: DeepEvalBaseLLM = None,
        include_reason: bool = True
    ):
        self.threshold = threshold
        self.model = model
        self.include_reason = include_reason
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase):
        # 1. Validate prerequisites
        if not test_case.retrieval_context:
            self.success = False
            self.score = 0.0
            self.reason = "Missing retrieval context."
            return self.score

        intermediate_steps = (
            test_case.additional_metadata.get("intermediate_steps", [])
            if test_case.additional_metadata else []
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

        # 3. Call the LLM Judge
        try:
            res = self.model.generate(prompt)
        except Exception as e:
            self.success = False
            self.score = 0.0
            self.reason = f"LLM generation failed: {str(e)}"
            return self.score

        # 4. Parse the response with robust error handling
        try:
            # Strip markdown formatting (e.g., ```json ... ```)
            clean_res = re.sub(r'^```json\s*|\s*```$', '', res.strip(), flags=re.MULTILINE)
            result_json = json.loads(clean_res)
            
            self.score = float(result_json.get("score", 0.0))
            self.reason = result_json.get("reason", "No reason provided by the judge.")
            self.success = self.score >= self.threshold
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.success = False
            self.score = 0.0
            self.reason = f"Failed to parse LLM judge response as JSON: {str(e)} | Raw response: {res[:200]}"
            
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Graph Multi-Hop Accuracy"