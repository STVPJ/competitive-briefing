"""Code-only filtering: URL dedup, content dedup and 7-day history tagging.

No model involved. Nothing here *drops* a story for being old news — tagging adds
fields the editor reads, and the editor decides.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

HISTORY_MAX_DAYS = 7

_STOP_WORDS = frozenset(
    "the a an and for in to of with on at by is are was were has have "
    "had its it this that from as or be".split()
)


def significant_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower())
            if w not in _STOP_WORDS and len(w) > 1}


def text_similarity(a: str, b: str) -> float:
    """Significant-word overlap over the mean word-set size (0.0–1.0)."""
    wa, wb = significant_words(a), significant_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / ((len(wa) + len(wb)) / 2)


def normalise_url(url: str) -> str:
    return (url or "").rstrip("/")


def deduplicate_by_url(results: list[dict]) -> list[dict]:
    """Keep the first copy of each URL.

    Order matters: feed results go first, so the copy kept carries the publisher's own
    date and outlet name rather than the search engine's.
    """
    seen, out = set(), []
    for r in results:
        url = normalise_url(r.get("url", ""))
        if url and url not in seen:
            seen.add(url)
            out.append(r)
    return out


def deduplicate_by_content(results: list[dict], threshold: float = 0.6) -> list[dict]:
    """Drop results whose headline overlaps an already-kept one above threshold,
    keeping the version with the longest snippet."""
    kept: list[dict] = []
    for r in sorted(results, key=lambda r: len(r.get("snippet", "")), reverse=True):
        if all(text_similarity(r.get("title", ""), k.get("title", "")) <= threshold for k in kept):
            kept.append(r)
    return kept


def load_history(path: Path, now: datetime | None = None) -> list[dict]:
    """Reported items from the last HISTORY_MAX_DAYS days."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("could not read history %s, starting fresh", path)
        return []
    cutoff = f"{(now or datetime.now()) - timedelta(days=HISTORY_MAX_DAYS):%Y-%m-%d}"
    return [item for item in data if item.get("date", "") >= cutoff]


def save_history(existing: list[dict], new_items: list[dict], path: Path,
                 now: datetime | None = None) -> None:
    cutoff = f"{(now or datetime.now()) - timedelta(days=HISTORY_MAX_DAYS):%Y-%m-%d}"
    pruned = [i for i in existing + new_items if i.get("date", "") >= cutoff]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pruned, indent=2))
    log.info("saved %d items to history (%d new)", len(pruned), len(new_items))


def tag_against_history(results: list[dict], history: list[dict]) -> list[dict]:
    """Tag each result "new" or "update" against the last 7 days. Never discards.

    An exact URL match, else the most similar headline above 0.5. The matched earlier
    report rides along as `prior_history`, so the editor compares against that one
    report instead of hunting through the 7-day list. Both sides are URL-normalised —
    stripping a trailing "/" on one side only meant identical articles never matched.
    """
    url_to_item = {}
    for h in history:
        for url in h.get("sources", []) if isinstance(h.get("sources"), list) else []:
            url_to_item[normalise_url(url)] = h

    for r in results:
        title = r.get("title", "").strip()
        match, how = url_to_item.get(normalise_url(r.get("url", ""))), "same_url"
        if match is None:
            best = max(((text_similarity(title, h.get("headline", "")), h) for h in history),
                       default=(0.0, None), key=lambda x: x[0])
            match, how = (best[1], "similar_headline") if best[0] > 0.5 else (None, "")
        r["tag"] = "update" if match else "new"
        if match:
            r["prior_history"] = {"reported_on": match.get("date", ""),
                                  "headline": match.get("headline", ""),
                                  "summary": match.get("summary", ""),
                                  "matched_by": how}
    return results
