from datetime import datetime, timedelta

from competitive_briefing import filters


def r(title, url, snippet=""):
    return {"title": title, "url": url, "snippet": snippet}


def test_url_dedup_keeps_first_copy_and_ignores_trailing_slash():
    feed = r("A", "https://x.example/a", "feed copy")
    search = r("A", "https://x.example/a/", "search copy")
    out = filters.deduplicate_by_url([feed, search])
    assert out == [feed]


def test_content_dedup_keeps_longest_snippet():
    short = r("Kestrel Code launches background agents to open pull requests", "https://a.example/1", "x")
    long = r("Kestrel Code launches background agents that open pull requests", "https://b.example/2", "x" * 50)
    other = r("Ploverline raises a Series B", "https://c.example/3", "y")
    out = filters.deduplicate_by_content([short, long, other])
    assert long in out and short not in out and other in out


def test_similarity_ignores_stop_words():
    assert filters.text_similarity("the launch of the agent", "launch agent") == 1.0
    assert filters.text_similarity("", "anything") == 0.0


def test_history_tagging_same_url_and_similar_headline():
    history = [{"date": "2026-10-19", "headline": "Tern Labs open-sources its sandbox runtime",
                "summary": "s", "sources": ["https://devwire.example/tern/"]}]
    results = [r("Totally different words", "https://devwire.example/tern"),
               r("Tern Labs open-sources sandbox runtime today", "https://other.example/x"),
               r("Kestrel ships agents", "https://k.example/1")]
    tagged = filters.tag_against_history(results, history)
    assert tagged[0]["tag"] == "update" and tagged[0]["prior_history"]["matched_by"] == "same_url"
    assert tagged[1]["tag"] == "update" and tagged[1]["prior_history"]["matched_by"] == "similar_headline"
    assert tagged[2]["tag"] == "new" and "prior_history" not in tagged[2]


def test_history_round_trip_prunes_old_items(tmp_path):
    now = datetime(2026, 10, 20)
    path = tmp_path / "h.json"
    old = {"date": f"{now - timedelta(days=10):%Y-%m-%d}", "headline": "old"}
    fresh = {"date": f"{now - timedelta(days=2):%Y-%m-%d}", "headline": "fresh"}
    filters.save_history([old], [fresh], path, now=now)
    assert [i["headline"] for i in filters.load_history(path, now=now)] == ["fresh"]
