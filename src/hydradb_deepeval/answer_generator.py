"""Generate an answer from retrieved context using an OpenAI-compatible LLM."""

from __future__ import annotations

import os

_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer the question using ONLY the provided context. "
    "Be concise and direct. If the context does not contain enough information, "
    "say so briefly."
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
