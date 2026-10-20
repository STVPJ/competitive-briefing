"""Minimal client for the Arctic Shift Reddit archive (https://arctic-shift.photon-reddit.com).

Unlike Reddit RSS, the archive carries `score` and `num_comments`, and it isn't
subject to Reddit's live rate limit. Engagement there matures with a lag of a day or
two, which is what the "gained traction" pass in reddit.py relies on.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger(__name__)

POSTS_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"
REQUEST_DELAY = 1.0   # seconds between requests
PAGE_LIMIT = 100      # Arctic Shift's per-request cap
MAX_RETRIES = 3       # on 429


def normalize_post(data: dict) -> dict | None:
    """Archive post → {subreddit, title, selftext, url, score, num_comments, created_utc}.

    None for deleted/removed posts or records missing an id or subreddit.
    """
    post_id, subreddit = data.get("id", ""), data.get("subreddit", "")
    if not post_id or not subreddit:
        return None
    selftext = data.get("selftext", "") or ""
    title = data.get("title", "")
    author = data.get("author", "") or ""
    if selftext in ("[deleted]", "[removed]"):
        return None
    if title == "[deleted]" and author in ("[deleted]", "[removed]"):
        return None
    return {
        "subreddit": subreddit,
        "title": title,
        "selftext": selftext,
        "url": f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/",
        "score": data.get("score", 0) or 0,
        "num_comments": data.get("num_comments", 0) or 0,
        "created_utc": int(data.get("created_utc", 0)),
    }


class ArcticShiftClient:
    def __init__(self, user_agent: str, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(headers={"User-Agent": user_agent}, timeout=30,
                                         follow_redirects=True, transport=transport)
        self._last = 0.0

    async def close(self) -> None:
        await self._client.aclose()

    async def _page(self, params: dict) -> list[dict]:
        for attempt in range(MAX_RETRIES + 1):
            wait = REQUEST_DELAY - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = await self._client.get(POSTS_URL, params=params)
                resp.raise_for_status()
                return resp.json().get("data", [])
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < MAX_RETRIES:
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                log.warning("arctic shift HTTP %s", e.response.status_code)
                return []
            except Exception as e:
                log.warning("arctic shift error: %s", e)
                return []
        return []

    async def fetch_subreddit_posts(self, subreddit: str, after: str, before: str) -> list[dict]:
        """All posts in a subreddit between two YYYY-MM-DD dates, paginating by date."""
        posts: list[dict] = []
        current = after
        while True:
            page = await self._page({"subreddit": subreddit, "after": current, "before": before,
                                     "limit": PAGE_LIMIT, "sort": "asc"})
            if not page:
                break
            posts.extend(page)
            last = datetime.fromtimestamp(page[-1].get("created_utc", 0), tz=timezone.utc)
            if f"{last:%Y-%m-%d}" <= current or len(page) < PAGE_LIMIT:
                break
            current = f"{last:%Y-%m-%d}"
        return posts
