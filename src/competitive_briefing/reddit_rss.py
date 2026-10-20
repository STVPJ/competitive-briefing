"""Reddit access via RSS/Atom feeds — no API key, no OAuth.

Reddit blocks anonymous access to its `.json` API (HTTP 403) and API-app creation
needs pre-approval, but the `.rss` feeds stay open. Trade-offs inherent to RSS: no
upvote score or comment count, ~25 items per feed.
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import httpx

ATOM = "{http://www.w3.org/2005/Atom}"
_ACCEPT = "application/atom+xml, application/xml, text/xml"
_SORTS = {"hot", "new", "top", "rising", "controversial"}
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# Reddit throttles .rss per IP by aggregate volume. A long backoff outlasts any
# sensible time budget, so fail fast with one short retry on an explicit 429.
_RETRY_STATUSES = {429}
_BACKOFF_SECONDS = (3,)


def _strip_html(raw: str) -> str:
    """Reddit Atom <content> is escaped HTML: unescape, drop tags, collapse whitespace."""
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", html.unescape(raw))
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def _to_epoch(iso: str) -> int:
    if not iso:
        return 0
    try:
        dt = datetime.fromisoformat(iso.strip().replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp())


def sanitize_subreddit(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_+]", "", re.sub(r"^/?r/", "", str(s or ""), flags=re.IGNORECASE))


def parse_atom(xml: str, limit: int = 25) -> list[dict]:
    """Reddit Atom feed → [{id, kind, title, author, link, subreddit, created_utc, text}].

    `kind` is 't3' (post) or 't1' (comment), from the Reddit fullname in <id>.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    entries = []
    for e in root.findall(f"{ATOM}entry"):
        def text(tag: str) -> str:
            node = e.find(f"{ATOM}{tag}")
            return node.text if node is not None and node.text else ""

        rid = text("id").strip()
        author = e.find(f"{ATOM}author/{ATOM}name")
        link = e.find(f"{ATOM}link")
        cat = e.find(f"{ATOM}category")
        entries.append({
            "id": rid,
            "kind": rid.split("_", 1)[0] if "_" in rid else "",
            "title": _strip_html(text("title")),
            "author": re.sub(r"^/?u/", "", (author.text or "") if author is not None else ""),
            "link": link.get("href", "") if link is not None else "",
            "subreddit": (cat.get("term") or "").strip() if cat is not None else "",
            "created_utc": _to_epoch(text("published") or text("updated")),
            "text": _strip_html(text("content")),
        })
        if len(entries) >= limit:
            break
    return entries


async def browse_subreddit(client: httpx.AsyncClient, subreddit: str, user_agent: str,
                           sort: str = "new", limit: int = 25) -> list[dict]:
    """A subreddit's posts via RSS. Raises httpx.HTTPStatusError on a non-200."""
    sub = sanitize_subreddit(subreddit)
    if not sub:
        raise ValueError("subreddit is required")
    url = f"https://www.reddit.com/r/{sub}/{sort if sort in _SORTS else 'new'}.rss"
    limit = max(1, min(int(limit), 25))
    for attempt in range(len(_BACKOFF_SECONDS) + 1):
        resp = await client.get(url, params={"limit": str(limit)},
                                headers={"User-Agent": user_agent, "Accept": _ACCEPT}, timeout=15)
        if resp.status_code == 200:
            return parse_atom(resp.text, limit)
        if resp.status_code not in _RETRY_STATUSES or attempt >= len(_BACKOFF_SECONDS):
            break
        await asyncio.sleep(_BACKOFF_SECONDS[attempt])
    raise httpx.HTTPStatusError(f"Reddit RSS returned HTTP {resp.status_code} for {url}",
                                request=resp.request, response=resp)
