"""Brave Web Search, freshness verification and article fetching.

Page fetches go through a `PageFetcher` (an async callable url -> HTML or None), so
the whole chain can run against bundled fixture pages with no network.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
RATE_LIMIT_DELAY = 1.5  # seconds between queries (the free plan allows 1/s)
MAX_QUERIES_PER_RUN = 15
RESULTS_PER_QUERY = 5

PageFetcher = Callable[[str], Awaitable[str | None]]


# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

def parse_brave_result(item: dict, query: str) -> dict:
    """A Brave web result → the common result shape."""
    extra = item.get("extra_snippets") or []
    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "snippet": item.get("description", ""),
        # page_age is Brave's own date for the page, which may be a re-crawl date.
        # verify_freshness() replaces it with the publisher's date where it can.
        "source_date": item.get("page_age") or item.get("age") or "",
        "extra_snippets": extra[:2],
        "query": query,
    }


async def run_searches(
    queries: list[str | dict],
    api_key: str | None,
    transport: httpx.AsyncBaseTransport | None = None,
    delay: float = RATE_LIMIT_DELAY,
) -> list[dict]:
    """Run Brave queries sequentially, rate limited. Returns combined results.

    An entry is a query string or a dict {"q": ..., plus extra Brave params such as
    "country", "search_lang", "freshness"} (used by regional editions).
    """
    if not api_key:
        log.warning("BRAVE_API_KEY not set — skipping web search")
        return []
    queries = queries[:MAX_QUERIES_PER_RUN]
    results: list[dict] = []
    async with httpx.AsyncClient(transport=transport, timeout=15) as client:
        for i, entry in enumerate(queries):
            if isinstance(entry, dict):
                extra = {k: v for k, v in entry.items() if k not in ("q", "freshness")}
                query, freshness = entry["q"], entry.get("freshness", "pd")
            else:
                query, freshness, extra = entry, "pd", {}
            try:
                resp = await client.get(
                    BRAVE_API_URL,
                    params={"q": query, "freshness": freshness, "count": RESULTS_PER_QUERY, **extra},
                    headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                )
                resp.raise_for_status()
                web = resp.json().get("web", {}).get("results", [])
                results += [parse_brave_result(item, query) for item in web]
                log.info("search [%d/%d] %r -> %d results", i + 1, len(queries), query, len(web))
            except httpx.HTTPStatusError as e:
                log.warning("search [%d/%d] %r -> HTTP %s", i + 1, len(queries), query,
                            e.response.status_code)
            except Exception as e:
                log.warning("search [%d/%d] %r -> error: %s", i + 1, len(queries), query, e)
            if i < len(queries) - 1:
                await asyncio.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Freshness verification
# ---------------------------------------------------------------------------
# A search engine's freshness filter and page date can be a re-crawl date: a
# year-old article came back dated "yesterday" and went out as news. Section and
# home pages always look fresh whatever story they carry. So every non-social result
# is fetched once and judged by the publisher's own date.

SOCIAL_DOMAINS = ("reddit.com", "news.ycombinator.com", "twitter.com", "x.com")
NON_ARTICLE_DOMAINS = ("wikipedia.org",)
MAX_ARTICLE_AGE_HOURS = 48

_PUBLISHED_RES = [
    re.compile(r'<meta[^>]+(?:property|name|itemprop)=["\'](?:article:published_time|'
               r'datePublished|pubdate|publishdate|date|DC\.date\.issued|parsely-pub-date|'
               r'sailthru\.date)["\'][^>]*content=["\']([^"\']+)', re.IGNORECASE),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name|itemprop)=["\']'
               r'(?:article:published_time|datePublished)["\']', re.IGNORECASE),
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
]
_OG_TYPE_RE = re.compile(
    r'<meta[^>]+property=["\']og:type["\'][^>]*content=["\']([^"\']+)', re.IGNORECASE)


def is_social(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith("." + d) for d in SOCIAL_DOMAINS)


def is_non_article(url: str, html: str | None = None) -> bool:
    """True for home pages, section/listing pages and encyclopaedia pages.

    URL alone only catches home pages and Wikipedia: URL depth is not a signal, because
    a one-segment path can be a real article (example.com/vendor-ships-new-agent).
    With the page, a section is JSON-LD CollectionPage, or og:type "website" on an
    address whose last segment is a plain word (example.com/technology). og:type alone
    isn't enough either: some publishers mark every article "website".
    """
    parsed = urlparse(url)
    if any(parsed.netloc.endswith(d) for d in NON_ARTICLE_DOMAINS):
        return True
    segments = [seg for seg in parsed.path.split("/") if seg]
    if not segments:
        return True
    if html is None:
        return False
    if re.search(r'"@type"\s*:\s*"CollectionPage"', html):
        return True
    og = _OG_TYPE_RE.search(html)
    # A plain word ("technology", "ai") reads as a section; slugs, ids and .html don't.
    section_like = re.fullmatch(r"[A-Za-z_]+", segments[-1]) is not None
    return bool(og and og.group(1).strip().lower() == "website" and section_like)


def extract_published(html: str) -> datetime | None:
    """The publisher's own publication date, from meta tags or JSON-LD."""
    for rx in _PUBLISHED_RES:
        for raw in rx.findall(html[:500_000]):
            try:
                dt = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            except ValueError:
                continue
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def extract_text(html: str) -> str:
    """Plain text from HTML: drop scripts/styles and tags, collapse whitespace."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                 ("&#39;", "'"), ("&nbsp;", " ")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def http_fetcher(user_agent: str, timeout: float = 8.0) -> tuple[PageFetcher, Callable[[], Awaitable[None]]]:
    """A PageFetcher over one shared httpx client, plus its close function."""
    client = httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                               headers={"User-Agent": user_agent})

    async def fetch(url: str) -> str | None:
        try:
            resp = await client.get(url)
        except Exception:
            return None
        return resp.text if resp.status_code == 200 else None

    return fetch, client.aclose


async def verify_freshness(
    results: list[dict],
    fetch: PageFetcher,
    max_age_hours: int = MAX_ARTICLE_AGE_HOURS,
    now: datetime | None = None,
) -> list[dict]:
    """Drop non-article pages and articles the publisher dates older than max_age_hours.

    Where the page declares a publication date it replaces the search engine's
    source_date (the publisher's date beats the search engine's). Where it declares
    none the result is kept on the search date — that is all there is. The page text
    is kept in `_text` so fetch_top_articles doesn't download it again.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    kept = [r for r in results if is_social(r.get("url", "")) or not is_non_article(r.get("url", ""))]
    for r in results:
        if r not in kept:
            log.info("dropped non-article page: %s", r.get("url", "")[:80])

    async def check(r: dict) -> None:
        page = await fetch(r["url"])
        if page is None:
            return
        if is_non_article(r["url"], page):
            r["_non_article"] = True
            return
        r["_text"] = extract_text(page)
        published = extract_published(page)
        if published:
            r["source_date"] = published.isoformat()
            r["_published"] = published

    await asyncio.gather(*(check(r) for r in kept if not is_social(r.get("url", ""))))

    fresh, undated = [], 0
    for r in kept:
        published = r.pop("_published", None)
        if r.pop("_non_article", False):
            log.info("dropped non-article page: %s", r["url"][:80])
            continue
        if published and published < cutoff:
            log.info("dropped stale article (published %s): %s", f"{published:%Y-%m-%d}", r["url"][:80])
            continue
        if not published and not is_social(r.get("url", "")):
            undated += 1
        fresh.append(r)
    log.info("freshness: kept %d/%d (%d with no publisher date, kept on search date)",
             len(fresh), len(results), undated)
    return fresh


async def fetch_top_articles(results: list[dict], fetch: PageFetcher, max_fetches: int = 8) -> list[dict]:
    """Add `full_text` (first 2,000 chars) to the most substantive non-social results.

    Snippet length is the proxy for substance. Social posts keep their snippet.
    """
    candidates = sorted((r for r in results if not is_social(r.get("url", ""))),
                        key=lambda r: len(r.get("snippet", "")), reverse=True)[:max_fetches]
    for r in candidates:
        text = r.get("_text")
        if text is None:
            page = await fetch(r["url"])
            text = extract_text(page) if page else ""
        if text:
            r["full_text"] = text[:2000]
    for r in results:
        r.pop("_text", None)
    return results
