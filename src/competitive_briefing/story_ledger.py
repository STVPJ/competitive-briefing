"""Permanent story ledger — long-horizon de-duplication.

The 7-day history (filters.py) is a *recency* signal. It cannot answer "have we
reported this story before?" for anything older, and in the original deployment that
is how one vendor's model launch was re-reported across 17 briefings in six weeks:
every new outlet writing about it arrived with a fresh URL and fresh headline wording,
so URL dedup, content dedup and the 7-day history all passed it through.

The ledger is keyed on the *story* rather than the *article*: a normalised entity set
plus an event class. "Acme ships its own coding model, Forge" and "Analysts weigh
Acme's Forge model" share little headline vocabulary, but both reduce to
`{acme, forge} + launch`-ish buckets that a candidate search can find.

Design constraints:
  - Keying, recording and the fallback `lookup()` are pure Python.
  - The "is this the same story?" judgement is made by TypeSafe Jev, one cheap request
    per (result, candidate) pair, when TYPESAFE_API_KEY is set. Without a key — or on
    any Jev failure — the word-overlap `lookup()` decides instead.
  - The ledger is NEVER injected into a prompt. A match attaches a small
    `prior_report` field to the one result it matched, nothing else reaches the model.
    A full-ledger injection would cost hundreds of thousands of input tokens per run
    and grow without bound.
  - Precision over recall. An uncertain match lets the story through, and nothing is
    dropped in code: `prior_report` is a hint to the editor, who still decides.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from . import jev

log = logging.getLogger(__name__)

# ~18 months: long enough that an annual cycle (earnings, conference keynotes) is
# still remembered; short enough that a genuinely revived story gets back through.
LEDGER_MAX_DAYS = 550
MAX_URLS_PER_STORY = 12

# Minimum text overlap for two mentions sharing a base key to count as the SAME story.
# It only discriminates *within* an already-matching entity+event bucket.
#
# The bar depends on how specific the bucket is, because a flat one provably fails.
# Measured over the original deployment's full archive (~2,500 story mentions):
#
#   threshold   stories   matches   catches the reported repeat?
#      0.20       1364      863     yes - but its weakest matches are plainly wrong
#      0.22       1551      648     yes
#      0.25       1764      402     NO
#      0.28       1893      269     NO
#
# The reported repeat scored 0.23, which a flat threshold can only catch by also
# admitting junk like two unrelated regulation stories in different countries. But
# that junk lives in GENERIC buckets (`unknown|...`, single-entity) while the real
# repeat lived in a two-entity `vendor+model|launch` bucket — specific enough that two
# genuinely different stories landing in it is unlikely. So specific buckets get a
# lower bar and generic ones a higher one, which separates the two instead of trading
# one against the other.
MERGE_SIMILARITY_SPECIFIC = 0.20   # base key names >= 2 entities
MERGE_SIMILARITY_GENERIC = 0.30    # base key names 0-1 entities


def merge_threshold(entities) -> float:
    """Similarity bar for a bucket, by how specific its entity set is."""
    return MERGE_SIMILARITY_SPECIFIC if len(entities) >= 2 else MERGE_SIMILARITY_GENERIC


# ---------------------------------------------------------------------------
# Entity vocabulary
# ---------------------------------------------------------------------------

# Players that show up constantly in AI news but aren't configured competitors.
# Only NAMED forms belong here. Generic phrases ("proprietary model", "in-house
# model") were tried and removed: they fire on any vendor's model and turn unrelated
# stories into false matches. Extend via the config's "entities" map.
BUILTIN_ENTITIES: dict[str, list[str]] = {
    "openai": ["OpenAI", "ChatGPT"],
    "anthropic": ["Anthropic"],
    "google": ["Google", "Gemini", "Google DeepMind", "DeepMind"],
    "microsoft": ["Microsoft", "Microsoft Copilot"],
    "meta": ["Meta", "Llama"],
    "mistral": ["Mistral", "Mistral AI"],
    "nvidia": ["Nvidia"],
    "eu-ai-act": ["EU AI Act", "AI Act", "European AI Act"],
}

_ENTITY_STOP = frozenset({"ai", "the", "new", "llm", "model"})

_VOCAB: list[tuple[str, str]] = []


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def build_vocabulary(competitors: list[dict], extra: dict[str, list[str]] | None = None) -> list[tuple[str, str]]:
    """(surface_form_lower, canonical_slug) pairs, longest surface form first.

    Config entries are added first and the first writer wins, so a competitor alias
    (e.g. bare "Copilot" for GitHub Copilot) can't be clobbered by a builtin.
    """
    vocab: dict[str, str] = {}

    def add(surface: str, canonical: str) -> None:
        s = surface.strip().lower()
        if len(s) >= 2:
            vocab.setdefault(s, canonical)

    for comp in competitors:
        # "Codex / OpenAI Codex" is two surface forms for one competitor.
        parts = [p.strip() for p in comp.get("name", "").split("/") if p.strip()]
        if not parts:
            continue
        canonical = slugify(parts[0])
        for p in parts:
            add(p, canonical)
            # "Foo AI" is usually written "Foo" in headlines. Only stem names long
            # enough to be unambiguous — short bare words fire on ordinary prose.
            stem = re.sub(r"\s+AI$", "", p, flags=re.I).strip()
            if stem != p and len(stem) >= 6:
                add(stem, canonical)
        for alias in comp.get("aliases") or []:
            add(alias, canonical)
    for source in (extra or {}, BUILTIN_ENTITIES):
        for canonical, surfaces in source.items():
            for s in surfaces or []:
                add(s, canonical)
    return sorted(vocab.items(), key=lambda kv: len(kv[0]), reverse=True)


def configure(competitors: list[dict], extra: dict[str, list[str]] | None = None) -> None:
    """Set the module vocabulary from config. Call once before keying stories."""
    global _VOCAB
    _VOCAB = build_vocabulary(competitors, extra)


def extract_entities(text: str) -> set[str]:
    """Canonical entity slugs in free text, longest surface form first.

    Every occurrence of a matched form is blanked out, so a longer form fully consumes
    its own text and a shorter overlapping form only fires on a genuinely separate
    mention. ("GitHub Copilot" is consumed before bare "Copilot" is tried.) Blanking
    only the first occurrence was a real bug: text naming a company twice leaked the
    shorter form and resolved it to a different entity.
    """
    if not text:
        return set()
    haystack = text.lower()
    found: set[str] = set()
    for surface, canonical in _VOCAB:
        pattern = r"(?<![a-z0-9])" + re.escape(surface) + r"(?![a-z0-9])"
        if not re.search(pattern, haystack):
            continue
        found.add(canonical)
        haystack = re.sub(pattern, lambda m: " " * (m.end() - m.start()), haystack)
    return {f for f in found if f not in _ENTITY_STOP}


# ---------------------------------------------------------------------------
# Event classification
# ---------------------------------------------------------------------------

# Most specific first: the first matching class wins. "Acquires X for $500M" is an
# acquisition, not funding, even though the money words also match funding.
_EVENT_CLASSES: list[tuple[str, str]] = [
    ("acquisition", r"\b(acquir\w*|acquisition|merger|merges|merged|buyout|takeover|buys|bought)\b"),
    # "raises guidance/outlook" is an earnings story wearing a funding verb.
    ("earnings_guidance", r"\b(raise[sd]?|raising|lift\w*|cut[s]?)\s+(its\s+|full-year\s+|fy\s+)*"
                          r"(guidance|outlook|forecast|target|estimates)\b"),
    ("funding", r"\b(raise[sd]?|raising|funding|series\s+[a-h]\b|valuation|valued|invest\w*|"
                r"backed|round|ipo|seed)\b"),
    ("legal_action", r"\b(sue[sd]?|suing|lawsuit|litigation|fined?|fine[sd]|sanction\w*|"
                     r"court|judge|ruling|ruled|settle\w*|verdict|penalt\w*)\b"),
    ("regulation", r"\b(regulat\w*|legislat\w*|bill|statute|watchdog|ban[s|ned]?|"
                   r"compliance|mandate[sd]?|ai act|enforcement|guidance|consultation)\b"),
    ("security", r"\b(vulnerab\w*|exploit\w*|breach\w*|leak\w*|cve|malicious|prompt injection)\b"),
    ("pricing", r"\b(pric\w*|subscription|plan[s]?|tier[s]?|rate limits?|free tier)\b"),
    ("partnership", r"\b(partner\w*|integrat\w*|teams? up|alliance|collaborat\w*|joins forces)\b"),
    ("personnel", r"\b(hire[sd]?|hiring|appoints?|appointed|names? new|steps down|"
                  r"resign\w*|departs?|layoffs?|lays off|promoted)\b"),
    ("launch", r"\b(launch\w*|unveil\w*|releas\w*|ship[s|ped]?|debut\w*|introduc\w*|"
               r"rolls? out|rollout|goes live|general availability|now available|"
               r"announc\w*|adds?|expand\w*|builds?|built|develop\w*)\b"),
    ("research", r"\b(report|study|studies|survey|benchmark|research|findings|"
                 r"analysis|data shows?)\b"),
    ("earnings", r"\b(earnings|quarterly|q[1-4]\b|revenue|results|growth)\b"),
]


def classify_event(text: str) -> str:
    low = (text or "").lower()
    for name, pattern in _EVENT_CLASSES:
        if re.search(pattern, low):
            return "earnings" if name == "earnings_guidance" else name
    return "other"


# ---------------------------------------------------------------------------
# Story keys and similarity
# ---------------------------------------------------------------------------

def story_key(headline: str) -> tuple[str, list[str], str]:
    """(key, entities, event_class), built from the HEADLINE only.

    Falling back to the summary was tried and removed: summaries name incidental
    parties (an investor in a funding round, say), and that one entry then acted as an
    attractor that wrongly matched a dozen unrelated stories. A headline states what
    the story IS. An entity-free headline keys as `unknown|<event>`, which only ever
    matches near-identical text — the safe failure.
    """
    ents = sorted(extract_entities(headline))
    event = classify_event(headline)
    return f"{'+'.join(ents) or 'unknown'}|{event}", ents, event


_SIM_STOP = frozenset(
    "the a an and for in to of with on at by is are was were has have had its it "
    "this that from as or be new own into now not just first it's".split()
)


def sig_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", (text or "").lower())
            if w not in _SIM_STOP and len(w) > 1}


def text_similarity(a: str, b: str) -> float:
    wa, wb = sig_words(a), sig_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / ((len(wa) + len(wb)) / 2)


def _entry_text(e: dict) -> str:
    return " ".join([e.get("headline", ""), e.get("summary", "")] + e.get("headlines", []))


# ---------------------------------------------------------------------------
# Persistence and recording
# ---------------------------------------------------------------------------

def load_ledger(path: Path, now: datetime | None = None) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.warning("could not read ledger %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        return {}
    cutoff = f"{(now or datetime.now()) - timedelta(days=LEDGER_MAX_DAYS):%Y-%m-%d}"
    return {k: v for k, v in data.items() if v.get("last_seen", "") >= cutoff}


def save_ledger(ledger: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
    log.info("saved %d stories to ledger", len(ledger))


def variant_keys(ledger: dict[str, dict], base_key: str) -> list[str]:
    """All stored keys for a base key: `base`, `base#2`, `base#3`, ..."""
    keys = [base_key] if base_key in ledger else []
    n = 2
    while f"{base_key}#{n}" in ledger:
        keys.append(f"{base_key}#{n}")
        n += 1
    return keys


def record_story(ledger: dict[str, dict], headline: str, summary: str = "",
                 urls: list[str] | None = None, date_str: str | None = None,
                 category: str = "") -> str:
    """Add or update one story. Returns its key.

    Two mentions sharing a base key merge only if their text also overlaps above the
    bucket's threshold; otherwise a `#n` variant is created. Without this, a coarse
    bucket like `vendor|launch` swallows every launch that vendor ever made — an early
    version collapsed ~2,500 mentions into ~300 "stories", destroying the first_seen
    dates the ledger exists to keep. `first_seen` only ever moves backwards.
    """
    date_str = date_str or f"{datetime.now():%Y-%m-%d}"
    base_key, ents, event = story_key(headline)
    text = f"{headline} {summary}"

    best_key, best_sim = None, 0.0
    for k in variant_keys(ledger, base_key):
        sim = text_similarity(text, _entry_text(ledger[k]))
        if sim > best_sim:
            best_key, best_sim = k, sim

    if best_key is None or best_sim < merge_threshold(ents):
        key, n = base_key, 2
        while key in ledger:
            key, n = f"{base_key}#{n}", n + 1
        ledger[key] = {
            "entities": ents, "event_class": event,
            "first_seen": date_str, "last_seen": date_str, "times_seen": 1,
            "headline": headline, "summary": summary, "category": category,
            "urls": list(urls or [])[:MAX_URLS_PER_STORY], "headlines": [headline],
        }
        return key

    entry = ledger[best_key]
    if date_str < entry.get("first_seen", date_str):
        entry.update(first_seen=date_str, headline=headline,
                     summary=summary or entry.get("summary", ""))
    entry["last_seen"] = max(entry.get("last_seen", date_str), date_str)
    entry["times_seen"] = entry.get("times_seen", 1) + 1
    known = entry.setdefault("headlines", [])
    if headline and headline not in known and len(known) < MAX_URLS_PER_STORY:
        known.append(headline)
    existing = entry.setdefault("urls", [])
    for u in urls or []:
        if u and u not in existing and len(existing) < MAX_URLS_PER_STORY:
            existing.append(u)
    return best_key


def build_entity_index(ledger: dict[str, dict]) -> dict[str, list[str]]:
    """entity slug -> [story keys], so a lookup touches a handful of entries."""
    index: dict[str, list[str]] = {}
    for key, entry in ledger.items():
        for ent in entry.get("entities", []):
            index.setdefault(ent, []).append(key)
    return index


def lookup(ledger: dict[str, dict], headline: str, summary: str = "") -> dict | None:
    """Word-overlap fallback: same entity set AND event class AND overlap over the
    bucket's threshold. All three, deliberately.

    A looser "same entities, any event class" tier was built and removed after
    measuring it: 237 matches at a median similarity of 0.11, mostly junk (a funding
    round matched to a forum thread that merely named the company). No similarity floor
    rescued it, because the genuine long-range catches scored 0.06–0.11, inside the junk
    band. Occasional duplicates getting through cost far less than real news being
    suppressed — nobody is going to audit what got blocked.
    """
    base_key, ents, _ = story_key(headline)
    text = f"{headline} {summary}"
    best = None
    for k in variant_keys(ledger, base_key):
        sim = text_similarity(text, _entry_text(ledger[k]))
        if sim >= merge_threshold(ents) and (best is None or sim > best[1]):
            best = (k, sim)
    if best is None:
        return None
    return {"key": best[0], "match": "overlap", "similarity": round(best[1], 2), **ledger[best[0]]}


# ---------------------------------------------------------------------------
# Jev same-story judgement
# ---------------------------------------------------------------------------
# `same_event` decides WHETHER it is a repeat; `relation` only labels a repeat as
# update vs rehash. On its own, relation's middle level soaks up uncertain, unrelated
# pairs, so it never gates.
JEV_SAME_EVENT = 0.8
JEV_REHASH = 1.5

# Candidates per result: exact-key variants + prior stories sharing any entity (top K
# by overlap) + the top K by overlap across the whole ledger, which catches stories
# the entity vocabulary misses. Every pool is capped: generic buckets hold hundreds of
# variants, and an uncapped first test sent 1.8M tokens for 14 results.
JEV_K_ENTITY = 5
JEV_K_TEXT = 3
JEV_MIN_TEXT_SIM = 0.10
JEV_SUMMARY_CHARS = 600

JEV_QUESTIONS = {
    "same_event": {
        "type": "noul",
        "instructions": "Do `new` and `prior` report the same news event?",
        "criteria": {
            "true": "Both describe the same specific announcement, launch, deal, funding "
                    "round, ruling, report or incident, even if worded differently, "
                    "published by different outlets, or `new` is a later article "
                    "about that same event.",
            "false": "They describe different events, even if they involve the same "
                     "companies, products or topic (for example two separate launches "
                     "by one company, or two different regulatory actions).",
        },
    },
    "relation": {
        "type": "score",
        "instructions": "How does the news reported in `new` relate to the news "
                        "reported in `prior`?",
        "criteria": [
            "`new` and `prior` report different events, even if they involve the same "
            "companies, products or topic (for example two separate launches by one "
            "company, or two different regulatory actions).",
            "`new` reports a later development of the event in `prior`: a new stage, "
            "outcome, date, figure or party that `prior` does not contain.",
            "`new` reports the same event as `prior` and adds nothing material: a "
            "restatement, recap, opinion piece or another outlet's coverage of the "
            "same facts.",
        ],
    },
}


def candidate_keys(ledger: dict, index: dict, eligible: set, words: dict,
                   headline: str, text: str) -> list[str]:
    base_key, ents, _ = story_key(headline)
    w_new = sig_words(text)

    def sim(k: str) -> float:
        wb = words[k]
        return len(w_new & wb) / ((len(w_new) + len(wb)) / 2) if w_new and wb else 0.0

    sims = {k: sim(k) for k in eligible}
    exact = sorted((k for k in variant_keys(ledger, base_key) if k in eligible),
                   key=lambda k: -sims[k])[:JEV_K_ENTITY]
    by_ent = sorted({k for e in ents for k in index.get(e, [])} & eligible,
                    key=lambda k: -sims[k])[:JEV_K_ENTITY]
    by_txt = [k for k in sorted(eligible, key=lambda k: -sims[k])[:JEV_K_TEXT]
              if sims[k] >= JEV_MIN_TEXT_SIM]
    return list(dict.fromkeys(exact + by_ent + by_txt))


def _age_days(entry: dict, today) -> int | None:
    try:
        return (today - datetime.strptime(entry["first_seen"], "%Y-%m-%d").date()).days
    except (ValueError, KeyError):
        return None


def jev_matches(results: list[dict], ledger: dict, index: dict, min_age_days: int,
                api_key: str, ask_many: Callable = jev.ask_many,
                now: datetime | None = None) -> dict[int, dict | None] | None:
    """Jev's best same-story match per result index; None overall if Jev is unusable.

    One request per (result, candidate) pair, never one per result: Jev loses accuracy
    as the state fills with irrelevant material. A result whose pairs did not ALL get
    an answer is left out of the map, so the caller falls back to `lookup()` for it
    rather than trusting a partial read.
    """
    today = (now or datetime.now()).date()
    ages = {k: _age_days(e, today) for k, e in ledger.items()}
    eligible = {k for k, age in ages.items() if age is not None and age >= min_age_days}
    words = {k: sig_words(_entry_text(ledger[k])) for k in eligible}

    pairs: list[tuple[int, str]] = []
    states: list[dict] = []
    for i, r in enumerate(results):
        title = r.get("title", "").strip()
        if not title:
            continue
        snippet = (r.get("snippet", "") or "")[:JEV_SUMMARY_CHARS]
        for k in candidate_keys(ledger, index, eligible, words, title, f"{title} {snippet}"):
            e = ledger[k]
            pairs.append((i, k))
            states.append({
                "new": {"headline": title, "summary": snippet},
                "prior": {"headline": e.get("headline", ""),
                          "summary": (e.get("summary", "") or "")[:JEV_SUMMARY_CHARS],
                          "first_reported": e.get("first_seen", "")},
            })
    if not states:
        return {}

    started = time.time()
    try:
        answers, tokens = ask_many(states, JEV_QUESTIONS, api_key)
    except Exception as e:
        log.warning("Jev unavailable (%s); using word-overlap lookup", e)
        return None
    failed = sum(1 for a in answers if a is None)
    log.info("Jev: %d pairs, %d input tokens, %.1fs, %d failed", len(states), tokens,
             time.time() - started, failed)
    if failed == len(answers):
        log.warning("all Jev requests failed; using word-overlap lookup")
        return None

    out: dict[int, dict | None] = {}
    broken: set[int] = set()
    for (i, k), a in zip(pairs, answers):
        if a is None:
            broken.add(i)
            continue
        same, rel = a["same_event"]["noul"], a["relation"]["score"]
        best = out.get(i)
        if same >= JEV_SAME_EVENT and (best is None or same > best["same_event"]):
            out[i] = {"key": k, "same_event": round(same, 2), "relation": round(rel, 2)}
        else:
            out.setdefault(i, None)
    for i in broken:
        out.pop(i, None)
    return out


def _audit(path: Path | None, title: str, hit: dict, age_days: int) -> None:
    """One line per flagged story, so a missing story is explainable after the fact.
    Nobody is expected to read it routinely."""
    if path is None:
        return
    why = (f"jev={hit['same_event']:.2f}\t{hit['assessment']}" if hit.get("match") == "jev"
           else f"sim={hit.get('similarity', 0):.2f}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{why}\tfirst={hit['first_seen']}"
                     f"\tage={age_days}d\tnow={title[:120]}\twas={hit.get('headline', '')[:120]}\n")
    except OSError:
        pass  # never let logging break a briefing


def tag_against_ledger(results: list[dict], ledger: dict, index: dict, min_age_days: int = 1,
                       jev_api_key: str | None = None, audit_path: Path | None = None,
                       ask_many: Callable = jev.ask_many, now: datetime | None = None) -> list[dict]:
    """Add `prior_report` to results that repeat a story first reported >= min_age_days ago.

    Does NOT discard. With a Jev key, Jev judges and the field carries an `assessment`
    ("rehash" or "update") as a hint; without one, the word-overlap `lookup()` decides
    and no assessment is given.
    """
    today = (now or datetime.now()).date()
    if jev_api_key:
        hits = jev_matches(results, ledger, index, min_age_days, jev_api_key, ask_many, now)
    else:
        log.info("TYPESAFE_API_KEY not set; story ledger uses the word-overlap fallback")
        hits = None

    tagged = via_jev = 0
    for i, r in enumerate(results):
        title = r.get("title", "").strip()
        if not title:
            continue
        if hits is not None and i in hits:
            j = hits[i]
            if not j:
                continue
            hit = {"key": j["key"], "match": "jev", "same_event": j["same_event"],
                   "assessment": "rehash" if j["relation"] >= JEV_REHASH else "update",
                   **ledger[j["key"]]}
            via_jev += 1
        else:
            hit = lookup(ledger, title, r.get("snippet", "") or "")
            if not hit:
                continue
        age = _age_days(hit, today)
        if age is None or age < min_age_days:
            continue
        r["prior_report"] = {
            "first_reported": hit["first_seen"],
            "days_ago": age,
            "times_reported": hit.get("times_seen", 1),
            "prior_headline": hit.get("headline", ""),
            "prior_summary": (hit.get("summary", "") or "")[:400],
        }
        if hit.get("match") == "jev":
            r["prior_report"]["assessment"] = hit["assessment"]
        tagged += 1
        _audit(audit_path, title, hit, age)

    log.info("ledger: %d result(s) match a story reported >= %d day(s) ago (%d judged by Jev)",
             tagged, min_age_days, via_jev)
    return results
