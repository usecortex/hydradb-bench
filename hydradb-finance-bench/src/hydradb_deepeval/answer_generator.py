"""Generate an answer from retrieved context using an OpenAI-compatible LLM."""

from __future__ import annotations

import os

_SYSTEM_PROMPT = (
    "You are a financial analyst specializing in interpreting corporate filings "
    "(10-K, 10-Q, 8-K) and financial statements (income statement, balance sheet, cash flow statement). "
    "Answer questions using ONLY the provided context. "
    "Follow these rules strictly:\n"
    "1. NUMBERS: Extract exact figures and express them in the units specified by the question "
    "(USD millions, USD billions, %, ratio, etc.). Round only when the question asks you to.\n"
    "2. CALCULATIONS: When the question requires a ratio, average, or derived metric, "
    "compute it using values from the context and show the result clearly.\n"
    "3. QUALITATIVE: For analysis questions (e.g. capital intensity, liquidity, margin drivers), "
    "support your conclusion with specific metrics or line items from the context.\n"
    "4. GROUNDING: Never introduce figures, assumptions, or knowledge outside the provided context."
)

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer:"""


async def generate_answer(
    question: str,
    context_str: str,
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call OpenAI chat completion to generate an answer grounded in context_str."""
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai package is required: pip install openai") from e

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OSError("OPENAI_API_KEY environment variable is not set")

    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    context=context_str,
                    question=question,
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""
