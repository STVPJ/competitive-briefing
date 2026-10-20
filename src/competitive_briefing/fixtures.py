"""Synthetic inputs for demo and test runs (`--fixtures`).

Everything in fixtures/synthetic.json is invented: fictional companies, `.example`
domains. Dates are stored as offsets ("hours_ago", "days_ago") and materialised
relative to now, so the demo always looks like today's news.

The fixture set is built to exercise each filter: a feed copy and a search copy of the
same URL, a near-duplicate headline, a re-crawled year-old article, a section page, a
home page, a repeat of yesterday's story and a repeat of a story from weeks ago.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from importlib import resources

from . import story_ledger
from .search import PageFetcher

FIXTURE_FILE = "synthetic.json"


def load(competitors: list[dict], entities: dict | None = None, now: datetime | None = None) -> dict:
    """Materialise the fixtures. Also configures the story-ledger vocabulary (config
    competitors + config entities + the fixtures' fictional companies), because the
    seeded ledger is keyed with it."""
    now = now or datetime.now(timezone.utc)
    raw = json.loads(resources.files(__package__).joinpath("fixtures", FIXTURE_FILE).read_text())
    story_ledger.configure(competitors, {**(entities or {}), **raw.get("entities", {})})

    def results(items: list[dict]) -> list[dict]:
        out = []
        for it in items:
            r = {k: v for k, v in it.items() if k != "hours_ago"}
            r.setdefault("extra_snippets", [])
            r["source_date"] = (now - timedelta(hours=it.get("hours_ago", 6))).isoformat()
            out.append(r)
        return out

    def day(n: int) -> str:
        return f"{now - timedelta(days=n):%Y-%m-%d}"

    history = [{**{k: v for k, v in h.items() if k != "days_ago"}, "date": day(h["days_ago"])}
               for h in raw["history"]]

    ledger: dict = {}
    for s in sorted(raw["ledger_stories"], key=lambda s: -s["days_ago"]):
        for n in range(s.get("times", 1)):
            story_ledger.record_story(ledger, s["headline"], s["summary"], s["urls"],
                                      day(s["days_ago"] - n), s["category"])

    pages = {}
    for url, p in raw["pages"].items():
        meta = []
        if p.get("published_days_ago") is not None:
            published = now - timedelta(days=p["published_days_ago"], hours=p.get("published_hours_ago", 0))
            meta.append(f'<meta property="article:published_time" content="{published.isoformat()}">')
        if p.get("og_type"):
            meta.append(f'<meta property="og:type" content="{p["og_type"]}">')
        if p.get("collection"):
            meta.append('<script type="application/ld+json">{"@type": "CollectionPage"}</script>')
        pages[url] = (f"<!doctype html><html><head>{''.join(meta)}</head>"
                      f"<body><article><p>{p.get('body', '')}</p></article></body></html>")

    return {
        "feeds": results(raw["feeds"]),
        "search": results(raw["search"]),
        "reddit": results(raw["reddit"]),
        "regional": {name: results(items) for name, items in raw.get("regional", {}).items()},
        "pages": pages,
        "history": history,
        "ledger": ledger,
    }


def page_fetcher(pages: dict[str, str]) -> PageFetcher:
    async def fetch(url: str) -> str | None:
        return pages.get(url)
    return fetch
