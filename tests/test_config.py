import json
from datetime import date

import pytest

from competitive_briefing import config as cfg
from competitive_briefing.render import format_date


def write(tmp_path, data):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(data))
    return p


MINIMAL = {
    "perspective": {"product": "P", "company": "C", "market": "M"},
    "editions": {"main": {"reader": "a PM on {product} at {company}", "reader_name": "R",
                          "sections": ["News"]}},
}


def test_bundled_config_loads_with_two_editions(config):
    assert list(config["editions"]) == ["main", "pt-br"]
    assert config["editorial"]["model"] == "claude-sonnet-5-5"
    assert config["editions"]["main"]["reader"] == "a Product Manager working on Acme Assist at Acme"


def test_defaults_and_substitution(tmp_path):
    c = cfg.load_config(write(tmp_path, MINIMAL))
    ed = c["editions"]["main"]
    assert ed["reader"] == "a PM on P at C"
    assert ed["why_label"] == "Why it matters:" and ed["include_global"] is True
    assert ed["name"] == "main"
    assert c["reddit"]["trending_min_score"] == 10


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.pop("editions"), "at least one edition"),
    (lambda d: d["perspective"].pop("market"), "perspective.market"),
    (lambda d: d["editions"]["main"].pop("sections"), "missing sections"),
    (lambda d: d.update(competitors=[{"aliases": []}]), "has no name"),
])
def test_validation_errors(tmp_path, mutate, message):
    data = json.loads(json.dumps(MINIMAL))
    mutate(data)
    with pytest.raises(cfg.ConfigError, match=message):
        cfg.load_config(write(tmp_path, data))


def test_missing_and_invalid_files(tmp_path):
    with pytest.raises(cfg.ConfigError, match="not found"):
        cfg.load_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(cfg.ConfigError, match="not valid JSON"):
        cfg.load_config(bad)


def test_global_queries_daily_only_and_year(config):
    q = cfg.global_queries(config, 2027)
    assert '"GitHub Copilot"' in q
    assert not any("Tabnine" in x for x in q)
    assert "AI code review funding OR acquisition 2027" in q
    assert len(q) <= 15


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    s = cfg.Settings()
    assert s.data_dir == tmp_path / "d" and s.typesafe_api_key is None


def test_localised_dates(config):
    d = date(2026, 10, 20)
    assert format_date(d, config["editions"]["main"]) == "Tuesday, 20 October 2026"
    assert format_date(d, config["editions"]["pt-br"]) == "terça-feira, 20 de outubro de 2026"
