from datetime import datetime

from competitive_briefing import story_ledger as sl

NOW = datetime(2026, 10, 20, 6, 0)


def test_longest_surface_form_consumes_shorter_one():
    # "Wren Cloud" is a different entity from bare "Wren" (an alias of Wren Studio):
    # the longer form consumes its own text, so Wren Studio does not fire.
    assert sl.extract_entities("Wren Cloud adds agents") == {"wren-cloud"}
    # A separate bare mention still resolves.
    assert sl.extract_entities("Wren Cloud vs Wren: a comparison") == {"wren-cloud", "wren-studio"}
    # Word boundaries: no match inside another word.
    assert sl.extract_entities("Kestrelfield council news") == set()


def test_config_alias_wins_over_builtin():
    # "Llama" is a builtin surface form; a competitor alias claims it first.
    vocab = dict(sl.build_vocabulary([{"name": "Starling", "aliases": ["Llama"]}]))
    assert vocab["llama"] == "starling"
    assert dict(sl.build_vocabulary([]))["llama"] == "meta"


def test_event_classes_are_ordered_specific_first():
    assert sl.classify_event("Kestrel acquires Ploverline for $500M") == "acquisition"
    assert sl.classify_event("Vendor raises guidance after strong quarter") == "earnings"
    assert sl.classify_event("Ploverline raises $40M Series B") == "funding"
    assert sl.classify_event("Tern CLI cuts price and adds a free tier") == "pricing"
    assert sl.classify_event("Kestrel launches background agents") == "launch"
    assert sl.classify_event("Nothing to see") == "other"


def test_story_key_uses_headline_only():
    key, ents, event = sl.story_key("Ploverline raises $40M Series B")
    assert key == "ploverline|funding" and ents == ["ploverline"] and event == "funding"
    assert sl.story_key("A startup raises money")[0] == "unknown|funding"


def test_threshold_depends_on_bucket_specificity():
    assert sl.merge_threshold(["a", "b"]) == sl.MERGE_SIMILARITY_SPECIFIC == 0.20
    assert sl.merge_threshold(["a"]) == sl.merge_threshold([]) == sl.MERGE_SIMILARITY_GENERIC == 0.30


def test_specific_bucket_merges_at_a_similarity_a_generic_bucket_rejects():
    # Two-entity bucket: overlap in [0.20, 0.30) merges.
    ledger = {}
    k1 = sl.record_story(ledger, "Kestrel Code and Ploverline partner on agent reviews",
                         "joint review workflow", date_str="2026-09-01")
    a = "Kestrel Code and Ploverline partner on agent reviews joint review workflow"
    b = "Kestrel and Ploverline integration deepens partnership with dashboards metrics pricing rollout"
    sim = sl.text_similarity(b, a)
    assert 0.20 <= sim < 0.30
    k2 = sl.record_story(ledger, "Kestrel and Ploverline integration deepens partnership",
                         "dashboards metrics pricing rollout", date_str="2026-09-10")
    assert k1 == k2 and ledger[k1]["times_seen"] == 2

    # One-entity bucket, same overlap band: a separate #2 story.
    ledger = {}
    k1 = sl.record_story(ledger, "Ploverline partners on agent reviews", "joint review workflow",
                         date_str="2026-09-01")
    k2 = sl.record_story(ledger, "Ploverline integration deepens partnership",
                         "dashboards metrics pricing rollout extras", date_str="2026-09-10")
    assert k1 != k2 and k2.endswith("#2")


def test_first_seen_only_moves_backwards():
    ledger = {}
    k = sl.record_story(ledger, "Ploverline raises $40M Series B", "led by Harbour", date_str="2026-09-10")
    sl.record_story(ledger, "Ploverline raises $40M Series B round", "led by Harbour", date_str="2026-09-01")
    sl.record_story(ledger, "Ploverline raises $40M Series B", "led by Harbour", date_str="2026-09-20")
    assert ledger[k]["first_seen"] == "2026-09-01" and ledger[k]["last_seen"] == "2026-09-20"


def test_tag_without_jev_key_falls_back_to_overlap(caplog):
    ledger = {}
    sl.record_story(ledger, "Ploverline raises $40M Series B for its AI code review agent",
                    "Ploverline raised a $40M Series B led by Harbour Capital", date_str="2026-08-20")
    results = [{"title": "Ploverline raises $40M Series B to take its code review agent to enterprises",
                "snippet": "a $40M Series B led by Harbour Capital"},
               {"title": "Kestrel launches agents", "snippet": "new"}]
    caplog.set_level("INFO")
    sl.tag_against_ledger(results, ledger, sl.build_entity_index(ledger), jev_api_key=None, now=NOW)
    assert "TYPESAFE_API_KEY not set" in caplog.text
    pr = results[0]["prior_report"]
    assert pr["first_reported"] == "2026-08-20" and pr["days_ago"] == 61
    assert "assessment" not in pr          # only Jev gives an assessment
    assert "prior_report" not in results[1]


def test_tag_with_jev_uses_pairwise_answers_and_assessment():
    ledger = {}
    sl.record_story(ledger, "Tern Labs open-sources its sandbox runtime", "Apache 2.0", date_str="2026-10-01")
    sl.record_story(ledger, "Kestrel launches agents", "background agents", date_str="2026-10-01")
    results = [{"title": "Tern Labs sandbox now open source", "snippet": "Apache licence"},
               {"title": "Kestrel launches agents in Europe", "snippet": "regional launch"}]
    seen_states = []

    def fake_ask_many(states, questions, api_key):
        seen_states.extend(states)
        answers = []
        for s in states:
            same = "Tern" in s["new"]["headline"] and "Tern" in s["prior"]["headline"]
            answers.append({"same_event": {"noul": 0.95 if same else 0.1},
                            "relation": {"score": 1.8 if same else 0.2}})
        return answers, 100 * len(states)

    sl.tag_against_ledger(results, ledger, sl.build_entity_index(ledger), jev_api_key="k",
                          ask_many=fake_ask_many, now=NOW)
    assert results[0]["prior_report"]["assessment"] == "rehash"
    assert "prior_report" not in results[1]
    # One request per (result, candidate) pair, each with exactly one prior story.
    assert all(set(s) == {"new", "prior"} for s in seen_states)


def test_jev_total_failure_falls_back():
    ledger = {}
    sl.record_story(ledger, "Ploverline raises $40M Series B", "Harbour Capital", date_str="2026-08-01")
    results = [{"title": "Ploverline raises $40M Series B", "snippet": "Harbour Capital"}]

    def broken(states, questions, api_key):
        return [None] * len(states), 0

    sl.tag_against_ledger(results, ledger, sl.build_entity_index(ledger), jev_api_key="k",
                          ask_many=broken, now=NOW)
    assert "prior_report" in results[0] and "assessment" not in results[0]["prior_report"]


def test_candidate_pools_are_capped():
    ledger = {}
    for n in range(40):
        sl.record_story(ledger, f"Something happened number {n} somewhere {n * 7}", f"detail {n}",
                        date_str="2026-09-01")
    index = sl.build_entity_index(ledger)
    words = {k: sl.sig_words(sl._entry_text(e)) for k, e in ledger.items()}
    keys = sl.candidate_keys(ledger, index, set(ledger), words, "Something happened somewhere",
                             "Something happened somewhere")
    assert len(keys) <= sl.JEV_K_ENTITY * 2 + sl.JEV_K_TEXT


def test_min_age_days_excludes_recent_stories():
    ledger = {}
    sl.record_story(ledger, "Ploverline raises $40M Series B", "Harbour Capital", date_str="2026-10-20")
    results = [{"title": "Ploverline raises $40M Series B", "snippet": "Harbour Capital"}]
    sl.tag_against_ledger(results, ledger, sl.build_entity_index(ledger), min_age_days=1, now=NOW)
    assert "prior_report" not in results[0]


def test_ledger_round_trip_prunes(tmp_path):
    path = tmp_path / "stories.json"
    ledger = {"a|other": {"last_seen": "2020-01-01"}, "b|other": {"last_seen": "2026-10-01"}}
    sl.save_ledger(ledger, path)
    assert list(sl.load_ledger(path, now=NOW)) == ["b|other"]
