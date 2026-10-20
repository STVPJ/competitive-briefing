"""End to end on the bundled synthetic fixtures, with the Anthropic client mocked."""

import json

import pytest

from competitive_briefing import pipeline
from competitive_briefing.config import ConfigError, Settings

from .conftest import CONFIG_PATH, FakeClient, briefing_input, tool_message


def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", output_dir=tmp_path / "output",
                    anthropic_api_key=None, brave_api_key=None, typesafe_api_key=None)


async def test_fixtures_dry_run_end_to_end(tmp_path):
    client = FakeClient(tool_message(briefing_input()), tool_message(briefing_input()))
    args = pipeline.parse_args(["--config", str(CONFIG_PATH), "--fixtures", "--dry-run"])
    paths = await pipeline.run(args, client=client, settings=settings(tmp_path))

    assert [p.name for p in paths][0].endswith(".html") and ".pt-br." in paths[1].name
    html = paths[0].read_text()
    assert "<h1>AI Coding Assistants" in html and "Kestrel Code launches background agents" in html
    assert paths[0].with_suffix(".md").read_text().startswith("# AI Coding Assistants")
    assert not (tmp_path / "data").exists()           # dry run: no state written

    # What the model was sent: the filters did their job.
    user = client.calls[0]["messages"][0]["content"]
    sent = json.loads(user.split("[UNTRUSTED_DATA_START]")[1].split("[UNTRUSTED_DATA_END]")[0])
    urls = [r["url"] for r in sent]
    assert "https://blogfarm.example/guide-ai-pair-programmers-2025" not in urls    # stale re-crawl
    assert "https://techdaily.example/ai" not in urls                                # section page
    assert "https://techdaily.example/" not in urls                                  # home page
    assert sum("kestrel" in u for u in urls) == 1                                    # URL + content dedup
    by_url = {r["url"]: r for r in sent}
    feed_copy = by_url["https://devwire.example/2026/kestrel-code-background-agents"]
    assert feed_copy["source"] == "DevWire" and feed_copy["full_text"]
    tern = by_url["https://devwire.example/2026/tern-sandbox-open-source"]
    assert tern["tag"] == "update" and tern["prior_history"]["matched_by"] == "same_url"
    plover = by_url["https://venturenotes.example/ploverline-series-b-enterprise"]
    assert plover["prior_report"]["times_reported"] == 2
    # The regional edition gets its own query results on top of the global set.
    pt_user = client.calls[1]["messages"][0]["content"]
    assert "Saíra" in pt_user and "Saíra" not in user


async def test_skip_compose_needs_no_api_key(tmp_path):
    args = pipeline.parse_args(["--config", str(CONFIG_PATH), "--fixtures", "--skip-compose",
                                "--edition", "main"])
    paths = await pipeline.run(args, settings=settings(tmp_path))
    data = json.loads(paths[0].read_text())
    assert "submit_briefing" in data["system"] and "[UNTRUSTED_DATA_START]" in data["user"]


async def test_compose_without_key_is_a_config_error(tmp_path):
    args = pipeline.parse_args(["--config", str(CONFIG_PATH), "--fixtures"])
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        await pipeline.run(args, settings=settings(tmp_path))


async def test_unknown_edition(tmp_path):
    args = pipeline.parse_args(["--config", str(CONFIG_PATH), "--fixtures", "--skip-compose",
                                "--edition", "fr"])
    with pytest.raises(ConfigError, match="unknown edition"):
        await pipeline.run(args, settings=settings(tmp_path))


async def test_real_run_writes_state(tmp_path, monkeypatch):
    """A non-dry run (sources stubbed) records history and the ledger for tomorrow."""
    async def fake_gather(config, s, fx):
        return [{"title": "Kestrel Code launches background agents", "url": "https://devwire.example/k",
                 "snippet": "s", "source_date": "2026-10-20T05:00:00+00:00", "query": "feed: DevWire"}]

    async def no_fetch(url):
        return None

    async def no_search(*a, **k):
        return []

    monkeypatch.setattr(pipeline, "gather_global", fake_gather)
    monkeypatch.setattr(pipeline.search, "http_fetcher", lambda ua: (no_fetch, None))
    monkeypatch.setattr(pipeline.search, "run_searches", no_search)
    client = FakeClient(tool_message(briefing_input()))
    args = pipeline.parse_args(["--config", str(CONFIG_PATH), "--edition", "main"])
    await pipeline.run(args, client=client, settings=settings(tmp_path))
    history = json.loads((tmp_path / "data" / "main" / "history.json").read_text())
    stories = json.loads((tmp_path / "data" / "main" / "stories.json").read_text())
    assert history[0]["headline"] == "Kestrel Code launches background agents"
    # Kestrel is not a configured competitor, so the story keys as unknown|launch.
    assert list(stories) == ["unknown|launch"]
