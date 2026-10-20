"""Configuration: one JSON file for *what* to watch, environment variables for *where*
state lives and for secrets.

The JSON config (default `config/briefing.json`) holds the competitive perspective,
competitors, feeds, subreddits and the editions. Secrets and paths come from the
environment (see `.env.example`), never from the config file, so the config can be
committed and shared.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config/briefing.json")

# Fields every edition must define. Everything else has a default (EDITION_DEFAULTS).
EDITION_REQUIRED = ("reader", "reader_name", "sections")

EDITION_DEFAULTS = {
    "title": "Competitive Briefing",
    "html_lang": "en",
    "why_label": "Why it matters:",
    "regulatory_scope": "",
    "extra_needs": "",
    "language_rule": "",
    "regional_queries": [],
    "include_global": True,
    "max_fetches": 8,
    "recipients": [],
    "subject_prefix": "",
    # Date heading. Names are in the config so non-English editions need no locale.
    "date_template": "{day_name}, {day} {month_name} {year}",
    "day_names": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                  "Sunday"],
    "month_names": ["January", "February", "March", "April", "May", "June", "July",
                    "August", "September", "October", "November", "December"],
}

EDITORIAL_DEFAULTS = {
    "model": "claude-sonnet-5-5",
    "effort": "medium",
    "max_tokens": 32000,
    "history_items_in_prompt": 30,
}


class ConfigError(ValueError):
    """The config file is missing something the pipeline needs."""


@dataclass
class Settings:
    """Runtime settings from the environment."""

    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "output")))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY") or None)
    brave_api_key: str | None = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY") or None)
    typesafe_api_key: str | None = field(default_factory=lambda: os.environ.get("TYPESAFE_API_KEY") or None)
    # Contact string sent in User-Agent headers. Feeds and Reddit ask for one.
    contact: str = field(default_factory=lambda: os.environ.get("BRIEFING_CONTACT", "competitive-briefing"))


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fill(text: str, perspective: dict) -> str:
    """Substitute {product} / {company} / {market} in a config string.

    Plain replace rather than str.format, so other braces in prompt text are safe.
    """
    for key in ("product", "company", "market"):
        text = text.replace("{" + key + "}", perspective.get(key, ""))
    return text


def load_config(path: Path | str | None = None) -> dict:
    """Load and validate the JSON config, filling defaults."""
    path = Path(path or os.environ.get("BRIEFING_CONFIG") or DEFAULT_CONFIG_PATH)
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"config file is not valid JSON: {path}: {e}") from e

    perspective = config.get("perspective") or {}
    for key in ("product", "company", "market"):
        if not perspective.get(key):
            raise ConfigError(f"perspective.{key} is required")

    competitors = config.setdefault("competitors", [])
    for n, comp in enumerate(competitors):
        if not comp.get("name"):
            raise ConfigError(f"competitors[{n}] has no name")
        comp.setdefault("aliases", [])
        comp.setdefault("search_terms", [])
        comp.setdefault("daily", True)

    config.setdefault("broad_queries", [])
    config.setdefault("news_feeds", [])
    config.setdefault("feed_keywords", [])
    config.setdefault("entities", {})
    reddit = config.setdefault("reddit", {})
    reddit.setdefault("subreddits", [])
    reddit.setdefault("keywords", [])
    reddit.setdefault("trending_lookback_days", 3)
    reddit.setdefault("trending_min_score", 10)
    reddit.setdefault("trending_min_comments", 8)
    config["editorial"] = {**EDITORIAL_DEFAULTS, **(config.get("editorial") or {})}

    editions = config.get("editions") or {}
    if not editions:
        raise ConfigError("at least one edition is required")
    for name, ed in editions.items():
        missing = [k for k in EDITION_REQUIRED if not ed.get(k)]
        if missing:
            raise ConfigError(f"edition {name!r} is missing {', '.join(missing)}")
        merged = {**EDITION_DEFAULTS, **ed, "name": name}
        for key, value in merged.items():
            if isinstance(value, str):
                merged[key] = _fill(value, perspective)
        editions[name] = merged
    config["editions"] = editions
    return config


def global_queries(config: dict, year: int) -> list[str]:
    """Competitor search terms (daily=true) + broad queries, `{year}` substituted."""
    queries = [t for c in config["competitors"] if c.get("daily", True) for t in c["search_terms"]]
    queries += [q.replace("{year}", str(year)) for q in config["broad_queries"]]
    return queries
