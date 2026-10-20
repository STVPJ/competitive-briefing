"""Publisher RSS/Atom feeds — the primary news source.

Feeds-first, search second. Web search was originally the only source, and over a
7-day sample only 2 of 33 trade-press articles reached the briefing that way, while
one-off low-tier sites filled the gaps. Feeds are free, carry the publisher's own
date (a search engine's date can be a re-crawl date), and name the outlet. Search
stays for competitor names and the long tail.

Aggregator feeds (Bing News / Google News RSS) were tried for competitor names and
rejected: one returned nothing or junk, the other returns encoded redirect links that
don't resolve to the article.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

log = logging.getLogger(__name__)

WINDOW_HOURS = 36          # same window as the Reddit fresh pass
FAIL_ALERT_RUNS = 3        # consecutive failed runs before a feed is flagged
QUIET_ALERT_DAYS = 14      # newest item older than this → feed flagged as gone quiet
SNIPPET_CHARS = 500

ATOM = "{http://www.w3.org/2005/Atom}"


def keyword_pattern(feed_keywords: list[str], competitors: list[dict]) -> re.Pattern | None:
    """One regex matching any feed keyword or competitor name/alias.

    Short all-caps terms (AI, LLM, IDE) match case-sensitively, so "ai" inside ordinary
    words and names doesn't count; everything else is case-insensitive.
    """
    terms = list(feed_keywords)
    for comp in competitors:
        terms += [p.strip() for p in comp.get("name", "").split("/") if p.strip()]
        terms += comp.get("aliases", [])
    parts = [rf"\b{re.escape(t)}\b" if t.isupper() and len(t) <= 4
             else rf"(?i:\b{re.escape(t)}\b)" for t in terms if t]
    return re.compile("|".join(parts)) if parts else None


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _clean(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(raw))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_date(raw: str) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom) → aware datetime; None if unparseable."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def looks_like_html(body: str) -> bool:
    """A bot wall serves a 200 HTML page, so test the body, not the status code."""
    return body.lstrip()[:200].lower().startswith(("<!doctype html", "<html"))


def parse_feed(xml_text: str) -> list[dict]:
    """RSS 2.0 or Atom → [{title, url, summary, published}] (published may be None)."""
    root = ET.fromstring(xml_text)
    items = []
    for it in root.iter("item"):
        items.append({
            "title": _clean(_text(it.find("title"))),
            "url": _text(it.find("link")),
            "summary": _clean(_text(it.find("description"))),
            "published": parse_date(_text(it.find("pubDate"))),
        })
    for it in root.iter(f"{ATOM}entry"):
        link = it.find(f"{ATOM}link[@rel='alternate']")
        if link is None:  # not `or`: an Element with no children is falsy
            link = it.find(f"{ATOM}link")
        items.append({
            "title": _clean(_text(it.find(f"{ATOM}title"))),
            "url": link.get("href", "") if link is not None else "",
            "summary": _clean(_text(it.find(f"{ATOM}summary")) or _text(it.find(f"{ATOM}content"))),
            "published": parse_date(_text(it.find(f"{ATOM}published"))
                                    or _text(it.find(f"{ATOM}updated"))),
        })
    return [i for i in items if i["title"] and i["url"].startswith("http")]


def update_health(status: dict[str, dict], path: Path | None) -> list[str]:
    """Record this run per feed; return alert lines for feeds that just crossed a threshold.

    Each problem is reported once (when it starts), not every day it persists.
    """
    health: dict = {}
    if path and path.exists():
        try:
            health = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            health = {}
    alerts, today = [], datetime.now().strftime("%Y-%m-%d")
    for name, st in status.items():
        h = health.setdefault(name, {"fails": 0, "flagged": None})
        if st["ok"]:
            h["fails"], h["last_ok"] = 0, today
            newest = st.get("newest")
            quiet = newest is not None and datetime.now(timezone.utc) - newest > timedelta(days=QUIET_ALERT_DAYS)
            if quiet and h.get("flagged") != "quiet":
                alerts.append(f"{name}: newest item is {newest:%Y-%m-%d} — feed may have moved or died")
                h["flagged"] = "quiet"
            elif not quiet:
                h["flagged"] = None
        else:
            h["fails"] += 1
            if h["fails"] >= FAIL_ALERT_RUNS and h.get("flagged") != "failing":
                alerts.append(f"{name}: failed {h['fails']} runs in a row ({st['error']})")
                h["flagged"] = "failing"
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(health, indent=2))
        except OSError as e:
            log.warning("could not save feed health: %s", e)
    return alerts


async def fetch_feed_items(
    feeds: list[dict],
    keyword_re: re.Pattern | None,
    user_agent: str,
    health_path: Path | None = None,
    window_hours: int = WINDOW_HOURS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[dict], list[str]]:
    """Fetch every feed; return (results in the common result shape, health alerts).

    A feed with `"filter": true` only contributes items matching keyword_re. Undated
    items are skipped: an undated feed item can't be shown to be news. Pass
    health_path=None to skip health tracking (dry runs, fixtures).
    """
    if not feeds:
        return [], []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    status: dict[str, dict] = {}
    results: list[dict] = []

    async with httpx.AsyncClient(timeout=15, follow_redirects=True, transport=transport,
                                 headers={"User-Agent": user_agent}) as client:
        async def one(feed: dict) -> None:
            name = feed["name"]
            try:
                # Per-feed UA override: some sites bot-wall every UA except a specific one.
                hdrs = {"User-Agent": feed["user_agent"]} if feed.get("user_agent") else None
                resp = await client.get(feed["url"], headers=hdrs)
                resp.raise_for_status()
                if looks_like_html(resp.text):
                    raise ValueError("got an HTML page, not a feed (bot wall?)")
                items = parse_feed(resp.text)
            except Exception as e:
                status[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:120]}
                log.warning("feed %s FAILED %s", name, status[name]["error"])
                return
            dated = [i["published"] for i in items if i["published"]]
            status[name] = {"ok": True, "newest": max(dated) if dated else None}
            kept = 0
            for i in items:
                if not i["published"] or i["published"] < cutoff:
                    continue
                if feed.get("filter") and keyword_re and not keyword_re.search(
                        f"{i['title']} {i['summary']}"):
                    continue
                results.append({
                    "title": i["title"],
                    "url": i["url"],
                    "snippet": i["summary"][:SNIPPET_CHARS],
                    "source_date": i["published"].isoformat(),
                    "extra_snippets": [],
                    "query": f"feed: {name}",
                    "source": name,
                })
                kept += 1
            log.info("feed %s: %d items, %d in window", name, len(items), kept)

        await asyncio.gather(*(one(f) for f in feeds))

    alerts = update_health(status, health_path) if health_path else []
    log.info("feeds: %d items from %d/%d feeds", len(results),
             sum(s["ok"] for s in status.values()), len(feeds))
    return results, alerts
