from datetime import datetime, timezone

import httpx
import pytest

from competitive_briefing import feeds, search

NOW = datetime(2026, 10, 20, 6, 0, tzinfo=timezone.utc)


def page(published=None, og=None, collection=False):
    meta = ""
    if published:
        meta += f'<meta property="article:published_time" content="{published}">'
    if og:
        meta += f'<meta property="og:type" content="{og}">'
    if collection:
        meta += '<script type="application/ld+json">{"@type": "CollectionPage"}</script>'
    return f"<html><head>{meta}</head><body><p>Body text.</p></body></html>"


@pytest.mark.parametrize("html,expected", [
    ('<meta property="article:published_time" content="2026-10-19T08:00:00Z">', "2026-10-19T08:00:00+00:00"),
    ('<meta content="2026-10-18T08:00:00+01:00" property="article:published_time">', "2026-10-18T08:00:00+01:00"),
    ('<script>{"datePublished": "2025-01-02"}</script>', "2025-01-02T00:00:00+00:00"),
    ("<p>no date</p>", None),
])
def test_extract_published(html, expected):
    got = search.extract_published(html)
    assert (got.isoformat() if got else None) == expected


@pytest.mark.parametrize("url,html,expected", [
    ("https://news.example/", None, True),                                        # home page
    ("https://en.wikipedia.org/wiki/Code_review", None, True),                    # encyclopaedia
    ("https://news.example/vendor-ships-agent", None, False),                     # one-segment article
    ("https://news.example/technology", page(og="website"), True),               # section page
    ("https://news.example/vendor-ships-agent", page(og="website"), False),      # og:type alone isn't enough
    ("https://news.example/2026/10/story.html", page(collection=True), True),    # CollectionPage
])
def test_is_non_article(url, html, expected):
    assert search.is_non_article(url, html) is expected


async def test_verify_freshness_publisher_date_beats_search_date():
    pages = {
        "https://a.example/fresh-story": page("2026-10-19T20:00:00Z", "article"),
        "https://a.example/recrawled-old-story": page("2025-09-01T00:00:00Z", "article"),
        "https://a.example/ai": page(og="website"),
        "https://a.example/no-date-story": page(og="article"),
    }

    async def fetch(url):
        return pages.get(url)

    results = [{"title": u, "url": u, "source_date": "2026-10-19"} for u in pages]
    results.append({"title": "home", "url": "https://a.example/", "source_date": "2026-10-20"})
    results.append({"title": "post", "url": "https://www.reddit.com/r/x/comments/1/", "source_date": "2026-10-20"})
    kept = await search.verify_freshness(results, fetch, now=NOW)
    urls = [r["url"] for r in kept]
    assert urls == ["https://a.example/fresh-story", "https://a.example/no-date-story",
                    "https://www.reddit.com/r/x/comments/1/"]
    assert kept[0]["source_date"] == "2026-10-19T20:00:00+00:00"   # replaced by the publisher's date
    assert kept[1]["source_date"] == "2026-10-19"                  # kept on the search date
    assert "_text" in kept[0]


async def test_fetch_top_articles_reuses_freshness_text():
    calls = []

    async def fetch(url):
        calls.append(url)
        return "<p>fetched</p>"

    results = [{"url": "https://a.example/1", "snippet": "long" * 10, "_text": "already have it"},
               {"url": "https://a.example/2", "snippet": "short"},
               {"url": "https://www.reddit.com/r/x/comments/1/", "snippet": "social" * 20}]
    out = await search.fetch_top_articles(results, fetch, max_fetches=8)
    assert out[0]["full_text"] == "already have it"
    assert out[1]["full_text"] == "fetched"
    assert "full_text" not in out[2] and calls == ["https://a.example/2"]
    assert all("_text" not in r for r in out)


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Wren adds agent mode</title><link>https://f.example/1</link>
<description>&lt;p&gt;Wren Studio ships agents.&lt;/p&gt;</description><pubDate>{recent}</pubDate></item>
<item><title>Gardening tips</title><link>https://f.example/2</link><description>Roses</description><pubDate>{recent}</pubDate></item>
<item><title>Old Wren story</title><link>https://f.example/3</link><description>x</description><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
<item><title>Undated Wren story</title><link>https://f.example/4</link><description>x</description></item>
</channel></rss>"""

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>An LLM release</title><link rel="alternate" href="https://g.example/1"/>
<updated>{recent}</updated><summary>s</summary></entry></feed>"""


def test_parse_feed_rss_and_atom():
    recent = "2026-10-19T10:00:00Z"
    items = feeds.parse_feed(ATOM.format(recent=recent))
    assert items[0]["url"] == "https://g.example/1" and items[0]["published"].year == 2026
    rss = feeds.parse_feed(RSS.format(recent="Mon, 19 Oct 2026 10:00:00 GMT"))
    assert rss[0]["summary"] == "Wren Studio ships agents."
    assert rss[3]["published"] is None


async def test_fetch_feed_items_filters_window_keywords_and_detects_html(tmp_path):
    recent = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

    def handler(request):
        if request.url.host == "f.example":
            return httpx.Response(200, text=RSS.format(recent=recent))
        if request.url.host == "wall.example":
            return httpx.Response(200, text="<!DOCTYPE html><html>Bot verification</html>")
        return httpx.Response(500)

    feed_list = [{"name": "F", "url": "https://f.example/rss", "filter": True},
                 {"name": "Wall", "url": "https://wall.example/rss"},
                 {"name": "Down", "url": "https://down.example/rss"}]
    pattern = feeds.keyword_pattern(["AI coding"], [{"name": "Wren Studio", "aliases": ["Wren"]}])
    health = tmp_path / "health.json"
    results, alerts = await feeds.fetch_feed_items(feed_list, pattern, "test-agent", health,
                                                   transport=httpx.MockTransport(handler))
    assert [r["url"] for r in results] == ["https://f.example/1"]
    assert results[0]["source"] == "F"
    assert alerts == []                                   # first failure isn't an alert yet
    for _ in range(2):
        _, alerts = await feeds.fetch_feed_items(feed_list, pattern, "test-agent", health,
                                                 transport=httpx.MockTransport(handler))
    assert len(alerts) == 2 and any("HTML page" in a for a in alerts)


def test_keyword_pattern_short_caps_are_case_sensitive():
    pat = feeds.keyword_pattern(["LLM", "coding agent"], [])
    assert pat.search("A new LLM for code")
    assert not pat.search("the llm-ish word")
    assert pat.search("Meet the Coding Agent")
