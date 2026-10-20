"""The pipeline, end to end, and the command line.

    gather (feeds → search → Reddit, once for all editions)
      → per edition: + regional search → URL dedup → content dedup → freshness check
      → 7-day history tagging → story-ledger tagging → full text for the top articles
      → ONE model call → source-link enforcement → files (+ email) → state

Usage:
    competitive-briefing                       # all editions, real sources
    competitive-briefing --edition main        # one edition
    competitive-briefing --dry-run             # compose + write output/, no email, no state writes
    competitive-briefing --fixtures --dry-run  # bundled synthetic inputs; only ANTHROPIC_API_KEY needed
    competitive-briefing --fixtures --skip-compose   # no API key at all: writes the model input instead
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

from . import delivery, feeds, filters, fixtures, reddit, render, search, story_ledger
from .config import ConfigError, Settings, global_queries, load_config, load_dotenv
from .editorial import EditorialError, build_system_prompt, build_user_message, compose_briefing, ensure_source_links

log = logging.getLogger("competitive_briefing")


async def gather_global(config: dict, settings: Settings, fx: dict | None) -> list[dict]:
    """Feeds first, then search, then Reddit. Order matters: URL dedup keeps the first
    copy, and the feed copy carries the publisher's own date and outlet name."""
    if fx is not None:
        return fx["feeds"] + fx["search"] + fx["reddit"]
    ua = f"competitive-briefing/0.1 (+{settings.contact})"
    feed_results, alerts = await feeds.fetch_feed_items(
        config["news_feeds"], feeds.keyword_pattern(config["feed_keywords"], config["competitors"]),
        ua, health_path=settings.data_dir / "feed-health.json")
    for a in alerts:
        log.warning("FEED ALERT: %s", a)
    search_results = await search.run_searches(global_queries(config, datetime.now().year),
                                                settings.brave_api_key)
    reddit_results = await reddit.fetch_reddit_posts(config["reddit"], ua)
    log.info("gathered: %d feed, %d search, %d reddit", len(feed_results), len(search_results),
             len(reddit_results))
    return feed_results + search_results + reddit_results


async def prepare(raw: list[dict], history: list[dict], ledger: dict, settings: Settings,
                  fetch: search.PageFetcher, max_fetches: int, audit_path: Path | None) -> list[dict]:
    """Everything between gather and compose. Pure code, no model."""
    results = filters.deduplicate_by_url(raw)
    log.info("URL dedup: %d -> %d", len(raw), len(results))
    n = len(results)
    results = filters.deduplicate_by_content(results)
    log.info("content dedup: %d -> %d", n, len(results))
    results = await search.verify_freshness(results, fetch)
    results = filters.tag_against_history(results, history)
    # min_age_days=1: the ledger also catches day-old repeats, in case the history
    # missed them (e.g. a day whose reported_items came back empty).
    results = story_ledger.tag_against_ledger(
        results, ledger, story_ledger.build_entity_index(ledger), min_age_days=1,
        jev_api_key=settings.typesafe_api_key, audit_path=audit_path)
    log.info("tagged: %d new, %d update, %d previously reported",
             sum(r.get("tag") == "new" for r in results),
             sum(r.get("tag") == "update" for r in results),
             sum("prior_report" in r for r in results))
    return await search.fetch_top_articles(results, fetch, max_fetches=max_fetches)


async def run_edition(name: str, edition: dict, config: dict, settings: Settings,
                      global_raw: list[dict], fx: dict | None, fetch: search.PageFetcher,
                      dry_run: bool, skip_compose: bool, primary: bool, client=None,
                      today: date | None = None) -> Path:
    today = today or date.today()
    state_dir = settings.data_dir / name
    if fx is not None:
        history, ledger = list(fx["history"]), json.loads(json.dumps(fx["ledger"]))
    else:
        history = filters.load_history(state_dir / "history.json")
        ledger = story_ledger.load_ledger(state_dir / "stories.json")
    log.info("[%s] history %d items, ledger %d stories", name, len(history), len(ledger))

    if fx is not None:
        regional = fx["regional"].get(name, [])
    else:
        regional = await search.run_searches(edition["regional_queries"], settings.brave_api_key)
    raw = regional + (global_raw if edition["include_global"] else [])
    audit = None if (dry_run or fx is not None) else state_dir / "dedup-audit.log"
    results = await prepare(raw, history, ledger, settings, fetch, edition["max_fetches"], audit)

    html_path, md_path = delivery.output_paths(settings.output_dir, name, today, primary)
    if skip_compose:
        path = html_path.with_suffix(".input.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "system": build_system_prompt(edition, config["perspective"], today),
            "user": build_user_message(results, history, today),
        }, indent=2, ensure_ascii=False))
        log.info("[%s] --skip-compose: model input written to %s", name, path)
        return path

    composed = await asyncio.to_thread(compose_briefing, results, history, edition,
                                       config["perspective"], config["editorial"], client, today)
    body, items = composed["briefing_html"], composed["reported_items"]
    if "<h3" in body.lower() and not items:
        log.warning("[%s] briefing has news items but reported_items is empty — tomorrow may "
                    "repeat them", name)
    body, _ = ensure_source_links(body, items, edition["why_label"])

    page = render.wrap_html(body, edition, today)
    delivery.write_files(page, render.to_markdown(body, edition, today), html_path, md_path)

    if dry_run or fx is not None:
        log.info("[%s] dry run: no state written, no email sent", name)
        return html_path

    # Durable state before the network send.
    filters.save_history(history, items, state_dir / "history.json")
    for item in items:
        story_ledger.record_story(ledger, item.get("headline", ""), item.get("summary", ""),
                                  item.get("sources") if isinstance(item.get("sources"), list) else [],
                                  today.isoformat(), item.get("category", ""))
    story_ledger.save_ledger(ledger, state_dir / "stories.json")

    recipients = edition["recipients"] or [r.strip() for r in os.environ.get("BRIEFING_RECIPIENTS", "").split(",") if r.strip()]
    if recipients and delivery.smtp_configured():
        subject = f"{edition['subject_prefix'] or edition['title']} — {render.format_date(today, edition)}"
        sent = delivery.send_email(subject, page, render.to_markdown(body, edition, today), recipients)
        log.info("[%s] emailed %d/%d recipient(s)", name, sent, len(recipients))
    return html_path


async def run(args: argparse.Namespace, client=None, settings: Settings | None = None) -> list[Path]:
    config = load_config(args.config)
    settings = settings or Settings()
    names = args.edition or list(config["editions"])
    unknown = [n for n in names if n not in config["editions"]]
    if unknown:
        raise ConfigError(f"unknown edition(s): {', '.join(unknown)}")
    if not args.skip_compose and not (settings.anthropic_api_key or client):
        raise ConfigError("ANTHROPIC_API_KEY is not set (use --skip-compose to run without it)")

    fx = None
    if args.fixtures:
        fx = fixtures.load(config["competitors"], config["entities"])
        fetch, close = fixtures.page_fetcher(fx["pages"]), None
    else:
        story_ledger.configure(config["competitors"], config["entities"])
        fetch, close = search.http_fetcher(f"Mozilla/5.0 (compatible; competitive-briefing/0.1; +{settings.contact})")

    try:
        global_raw = await gather_global(config, settings, fx)
        first = next(iter(config["editions"]))
        paths = []
        for name in names:
            # Each edition gets its own copy: tagging annotates results in place.
            paths.append(await run_edition(
                name, config["editions"][name], config, settings, json.loads(json.dumps(global_raw)),
                fx, fetch, args.dry_run, args.skip_compose, primary=(name == first), client=client))
        return paths
    finally:
        if close:
            await close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="competitive-briefing", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="config JSON (default: $BRIEFING_CONFIG or config/briefing.json)")
    p.add_argument("--edition", action="append", help="edition to run (repeatable; default: all)")
    p.add_argument("--dry-run", action="store_true", help="compose and write output/, but no email and no state writes")
    p.add_argument("--fixtures", action="store_true", help="use bundled synthetic inputs instead of the network")
    p.add_argument("--skip-compose", action="store_true", help="stop before the model call; write its input as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        paths = asyncio.run(run(args))
    except (ConfigError, EditorialError) as e:
        log.error("%s", e)
        return 1
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
