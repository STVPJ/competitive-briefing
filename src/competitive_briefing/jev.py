"""Thin client for TypeSafe's System One API (Jev).

Jev returns typed judgements (Noul / Choice / Score) about a `state` rather than
generated text, and is priced per input token only. Docs: https://docs.typesafe.ai

Stdlib only (urllib). Callers must treat any exception or `None` answer as "Jev
unavailable" and fall back to their deterministic path: a Jev outage must never break
a briefing.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API_URL = "https://api.typesafe.ai/v1/systemone"
# Pinned, not `jev-latest`: a silent model bump would move the calibrated thresholds
# in story_ledger (JEV_SAME_EVENT, JEV_REHASH), which were measured on this version.
MODEL = "jev-1.13.0"

# The public endpoint rate-limits above roughly eight concurrent requests.
WORKERS = 6


def ask(state, questions: dict, api_key: str, timeout: float = 20, retries: int = 2) -> dict:
    """One request. Returns the raw response (`answers`, `usage`). Raises on failure."""
    body = json.dumps({"state": state, "model": MODEL, "questions": questions}).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(API_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 429 and e.code < 500:
                raise
            time.sleep(min(float(e.headers.get("retry-after") or 2 ** attempt), 10))
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Jev request failed after {retries + 1} attempts: {last}")


def ask_many(states: list, questions: dict, api_key: str) -> tuple[list[dict | None], int]:
    """Ask the same questions about many states in parallel.

    Returns (answers per state — None where that request failed, total input tokens).
    """
    def one(state):
        try:
            data = ask(state, questions, api_key)
            return data["answers"], data.get("usage", {}).get("input_tokens", 0)
        except Exception:
            return None, 0

    if not states:
        return [], 0
    with ThreadPoolExecutor(min(WORKERS, len(states))) as pool:
        results = list(pool.map(one, states))
    return [a for a, _ in results], sum(t for _, t in results)
