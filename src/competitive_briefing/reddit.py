"""Reddit practitioner chatter, in two passes.

1. Fresh (last 36 h) via Reddit RSS — recency, but no engagement data.
2. "Gained traction" via the Arctic Shift archive — posts from the last few days
   that have since accumulated real engagement. Delayed, but a genuine signal, and
   the best available now that fresh-post upvotes aren't readable without auth.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx

from . import reddit_rss
from .arctic_shift import ArcticShiftClient, normalize_post

log = logging.getLogger(__name__)

RATE_LIMIT_DELAY = 2.0   # seconds between subreddit feeds
MAX_AGE_HOURS = 36       # slightly wider than 24 h for timezone edges


def keyword_patterns(keywords: list[str]) -> list[re.Pattern]:
    return [re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE) for k in keywords]


def is_relevant(title: str, body: str, patterns: list[re.Pattern]) -> bool:
    """Whole-word keyword match. No keywords configured → everything is relevant."""
    if not patterns:
        return True
    text = f"{title} {body}"
    return any(p.search(text) for p in patterns)


def _result(subreddit: str, title: str, body: str, url: str, created: int, query: str,
            score: int = 0, comments: int = 0) -> dict:
    return {
        "title": f"[r/{subreddit}] {title}",
        "url": url,
        "snippet": body[:500] if body else title,
        "source_date": datetime.fromtimestamp(created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extra_snippets": [],
        "query": query,
        "source": "reddit",
        "score": score,
        "num_comments": comments,
    }


def parse_rss_post(entry: dict, patterns: list[re.Pattern], now: datetime | None = None) -> dict | None:
    """RSS entry → result, or None if it's a comment, too old or off-topic."""
    if entry.get("kind") != "t3":
        return None
    now = now or datetime.now(timezone.utc)
    created = entry.get("created_utc", 0)
    if datetime.fromtimestamp(created, tz=timezone.utc) < now - timedelta(hours=MAX_AGE_HOURS):
        return None
    if not is_relevant(entry.get("title", ""), entry.get("text", ""), patterns):
        return None
    sub = entry.get("subreddit", "")
    return _result(sub, entry.get("title", ""), entry.get("text", ""), entry.get("link", ""),
                   created, f"reddit:r/{sub}")


async def fetch_fresh(subreddits: list[str], patterns: list[re.Pattern], user_agent: str,
                      transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    out: list[dict] = []
    async with httpx.AsyncClient(follow_redirects=True, transport=transport) as client:
        for i, sub in enumerate(subreddits):
            client.cookies.clear()
            try:
                entries = await reddit_rss.browse_subreddit(client, sub, user_agent, limit=25)
            except Exception as e:
                log.warning("r/%s: %s", sub, e)
                entries = []
            posts = [p for p in (parse_rss_post(e, patterns) for e in entries) if p]
            log.info("r/%s: %d posts, %d relevant in last %dh", sub, len(entries), len(posts), MAX_AGE_HOURS)
            out += posts
            if i < len(subreddits) - 1:
                await asyncio.sleep(RATE_LIMIT_DELAY)
    return out


async def fetch_trending(subreddits: list[str], patterns: list[re.Pattern], user_agent: str,
                         lookback_days: int, min_score: int, min_comments: int,
                         transport: httpx.AsyncBaseTransport | None = None) -> list[dict]:
    """Relevant posts from the last few days with score >= min_score OR comments >= min_comments."""
    now = datetime.now(timezone.utc)
    after = f"{now - timedelta(days=lookback_days):%Y-%m-%d}"
    before = f"{now + timedelta(days=1):%Y-%m-%d}"
    client = ArcticShiftClient(user_agent, transport=transport)
    out: list[dict] = []
    try:
        for sub in subreddits:
            for raw in await client.fetch_subreddit_posts(sub, after, before):
                post = normalize_post(raw)
                if not post or not is_relevant(post["title"], post["selftext"], patterns):
                    continue
                if post["score"] < min_score and post["num_comments"] < min_comments:
                    continue
                out.append(_result(post["subreddit"], post["title"], post["selftext"], post["url"],
                                   post["created_utc"], "reddit:trending",
                                   post["score"], post["num_comments"]))
    finally:
        await client.close()
    log.info("reddit trending: %d posts", len(out))
    return out


def merge(fresh: list[dict], trending: list[dict]) -> list[dict]:
    """Dedup by URL — trending wins, since it carries real engagement — then
    engagement first (fresh RSS posts score 0)."""
    by_url = {p["url"]: p for p in fresh + trending}
    return sorted(by_url.values(), key=lambda r: r.get("score", 0), reverse=True)


async def fetch_reddit_posts(reddit_config: dict, user_agent: str) -> list[dict]:
    subs = reddit_config.get("subreddits", [])
    if not subs:
        return []
    patterns = keyword_patterns(reddit_config.get("keywords", []))
    fresh = await fetch_fresh(subs, patterns, user_agent)
    trending = await fetch_trending(subs, patterns, user_agent,
                                    reddit_config["trending_lookback_days"],
                                    reddit_config["trending_min_score"],
                                    reddit_config["trending_min_comments"])
    return merge(fresh, trending)
