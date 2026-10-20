from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from competitive_briefing import story_ledger
from competitive_briefing.config import load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "briefing.json"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture(autouse=True)
def vocabulary():
    """A small, predictable entity vocabulary for every test (fictional names only)."""
    story_ledger.configure(
        [{"name": "Wren Studio", "aliases": ["Wren"]},
         {"name": "Kestrel Code", "aliases": ["Kestrel"]},
         {"name": "Ploverline", "aliases": []}],
        {"tern-labs": ["Tern Labs"], "wren-cloud": ["Wren Cloud"]})
    yield


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeClient:
    """Stands in for anthropic.Anthropic: records each request, replays canned messages."""

    def __init__(self, *messages):
        self.queue = list(messages)
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.queue.pop(0))


def tool_message(tool_input: dict, stop_reason: str = "tool_use"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=30_000, output_tokens=8_000),
        content=[SimpleNamespace(type="tool_use", name="submit_briefing", input=tool_input)],
    )


def text_message(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=30_000, output_tokens=500),
        content=[SimpleNamespace(type="text", text=text)],
    )


def briefing_input(today: str = "2026-10-20") -> dict:
    return {
        "repeat_check": [],
        "reported_items": [{
            "date": today, "headline": "Kestrel Code launches background agents",
            "summary": "Kestrel's agents open pull requests on their own.",
            "sources": ["https://devwire.example/2026/kestrel-code-background-agents"],
            "category": "product"}],
        "briefing_html": ("<p>One launch matters today.</p><h2>Product &amp; Feature News</h2>"
                          "<h3>Kestrel Code launches background agents</h3><p>Summary.</p>"
                          "<p><strong>Why it matters:</strong> Direct threat.</p>"),
    }


@pytest.fixture
def fake_client_factory():
    return FakeClient


def dump(obj) -> str:
    return json.dumps(obj, indent=2)
