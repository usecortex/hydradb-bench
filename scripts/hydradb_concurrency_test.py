"""
Quick concurrency test for HydraDB full_recall endpoint.
Fires N concurrent requests and measures success rate + latency.

Usage:
    python scripts/hydradb_concurrency_test.py
"""
import asyncio
import os
import statistics
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL   = os.environ["HYDRADB_BASE_URL"]
API_KEY    = os.environ["HYDRADB_API_KEY"]
TENANT_ID  = os.environ["HYDRADB_TENANT_ID"]
SUB_TENANT = "legal-privacyqa"

# A few representative questions from PrivacyQA
QUESTIONS = [
    "Does this policy allow sharing user data with third parties?",
    "What personal information is collected by this service?",
    "How long is user data retained?",
    "Can users request deletion of their data?",
    "Is location data collected and stored?",
    "What security measures protect user information?",
    "Are cookies used for tracking purposes?",
    "Can users opt out of data collection?",
    "Is user data sold to advertisers?",
    "What happens to data when an account is deleted?",
]

LEVELS = [1, 3, 5, 10, 15, 20]
MODES  = ["fast", "thinking"]


async def single_request(client: httpx.AsyncClient, question: str, mode: str = "fast") -> tuple[float, bool]:
    payload = {
        "tenant_id": TENANT_ID,
        "sub_tenant_id": SUB_TENANT,
        "query": question,
        "max_results": 5,
        "graph_context": True,
        "mode": mode,
    }
    t = time.monotonic()
    try:
        r = await client.post(
            "/recall/full_recall",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        elapsed = (time.monotonic() - t) * 1000
        ok = r.status_code < 400
        if not ok:
            print(f"    HTTP {r.status_code}: {r.text[:120]}")
        return elapsed, ok
    except Exception as e:
        elapsed = (time.monotonic() - t) * 1000
        print(f"    Exception: {e}")
        return elapsed, False


async def run_level(concurrency: int, mode: str = "fast") -> None:
    questions = (QUESTIONS * 10)[:concurrency]

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        tasks = [single_request(client, q, mode) for q in questions]
        wall_start = time.monotonic()
        results = await asyncio.gather(*tasks)
        wall_ms = (time.monotonic() - wall_start) * 1000

    latencies = [r[0] for r in results]
    successes = sum(1 for r in results if r[1])
    failures  = concurrency - successes

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    mean = statistics.mean(latencies)

    status = "OK" if failures == 0 else f"FAIL ({failures} errors)"
    print(
        f"  concurrency={concurrency:>3} | "
        f"success={successes}/{concurrency} | "
        f"mean={mean:>7.0f}ms | p50={p50:>7.0f}ms | p95={p95:>7.0f}ms | "
        f"wall={wall_ms:>7.0f}ms | {status}"
    )


async def main() -> None:
    print(f"HydraDB Concurrency Test")
    print(f"Endpoint : {BASE_URL}/recall/full_recall")
    print(f"Tenant   : {TENANT_ID} / {SUB_TENANT}")
    print("=" * 85)
    for mode in MODES:
        print(f"\nMode: {mode.upper()}  |  max_results=5  |  graph_context=True")
        print("-" * 85)
        for level in LEVELS:
            await run_level(level, mode)
            await asyncio.sleep(1)
    print("=" * 85)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
