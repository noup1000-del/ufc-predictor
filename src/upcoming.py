"""Upcoming UFC cards.

Sources, in order: a hand-made `data/raw/upcoming_card.json` (manual override), the official
schedule at ufc.com/events (each event page parsed into `data/raw/cards/<date>_<slug>.json`,
and the next card synced to `upcoming_card.json`), then ESPN's scoreboard API. If every
source refuses (403 / challenge page), fail clearly and ask for the manual file.

Fighter names are mapped to fighter_id with `src.matching`; unmatched names stay
visible (fighter_id None) rather than being guessed.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from src.config import load_config, resolve_path
from src.http import FetchError, HttpClient, get_client
from src.matching import FighterMatcher, canonical_weight_class, normalize_name

logger = logging.getLogger(__name__)

# Events on ESPN's UFC scoreboard that are not UFC cards (no ufcstats history / not predicted).
_NON_UFC_EVENT_MARKERS = ("contender series", "road to ufc", "ultimate fighter")


class UpcomingCardError(Exception):
    """No upcoming card could be determined."""


@dataclass
class Bout:
    fighter_1: str
    fighter_2: str
    weight_class: str | None = None
    bout_order: int | None = None
    fighter_1_id: str | None = None
    fighter_2_id: str | None = None
    fighter_1_match: str | None = None  # how the id was found (manual/exact/fuzzy/...)
    fighter_2_match: str | None = None
    is_title_fight: int | None = None   # None -> predict.py default (0)
    scheduled_rounds: int | None = None  # None -> predict.py default (5 for bout 1, else 3)


@dataclass
class Card:
    event_name: str
    event_date: str  # YYYY-MM-DD
    source: str      # "manual" | "espn" | "ufc.com"
    bouts: list[Bout] = field(default_factory=list)
    location: str | None = None
    event_url: str | None = None
    fetched_at: str | None = None   # when the card was last read from the source (UTC ISO)
    # Line-up changes seen between fetches, oldest first (see `diff_bouts`), each with `detected_at`.
    changes: list[dict] = field(default_factory=list)

    def unmatched(self) -> list[str]:
        out = []
        for b in self.bouts:
            if not b.fighter_1_id:
                out.append(b.fighter_1)
            if not b.fighter_2_id:
                out.append(b.fighter_2)
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(b) for b in self.bouts])


# --------------------------------------------------------------------------- manual override

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_manual_card(path: Path, today: date) -> Card | None:
    """Read a card file (manual, or written by the ufc.com fetch); None if its date has passed."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        event_date = date.fromisoformat(data["event_date"])
        bouts = [
            Bout(fighter_1=b["fighter_1"], fighter_2=b["fighter_2"],
                 weight_class=canonical_weight_class(b.get("weight_class")) or b.get("weight_class"),
                 bout_order=b.get("bout_order", i + 1),
                 fighter_1_id=b.get("fighter_1_id") or None, fighter_2_id=b.get("fighter_2_id") or None,
                 fighter_1_match="manual" if b.get("fighter_1_id") else None,
                 fighter_2_match="manual" if b.get("fighter_2_id") else None,
                 is_title_fight=b.get("is_title_fight"), scheduled_rounds=b.get("scheduled_rounds"))
            for i, b in enumerate(data["bouts"])
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise UpcomingCardError(f"invalid manual card file {path}: {e}") from e
    if event_date < today:
        logger.warning("Manual card %s is dated %s (in the past); ignoring it", path, event_date)
        return None
    return Card(event_name=data.get("event_name", "Manual card"), event_date=event_date.isoformat(),
                source=data.get("source") or "manual", bouts=bouts, location=data.get("location"),
                event_url=data.get("event_url"), fetched_at=data.get("fetched_at"),
                changes=list(data.get("changes") or []))


# --------------------------------------------------------------------------- ufc.com

@dataclass
class ScheduledEvent:
    """One entry of the upcoming list on ufc.com/events."""
    url: str
    headline: str          # e.g. "Rosas Jr. vs Barcelos"
    event_date: str        # YYYY-MM-DD (US Eastern date, as ufc.com displays it)
    location: str | None


def parse_ufc_events_page(html: str, base_url: str) -> list[ScheduledEvent]:
    """Upcoming events (the `upcoming` view only; past results are ignored), in date order.

    Each `.c-card-event--result` card gives the event link and headline, a displayed date
    without the year ("Sat, Sep 26 / 8:00 PM EDT") and a Unix timestamp that supplies it.
    """
    soup = BeautifulSoup(html, "lxml")
    view = soup.select_one(".view-display-id-upcoming")
    if view is None:
        raise UpcomingCardError("ufc.com events page has no upcoming-events list (layout changed?)")
    events, seen = [], set()
    for card in view.select(".c-card-event--result"):
        link = card.select_one(".c-card-event--result__headline a")
        date_el = card.select_one(".c-card-event--result__date")
        if link is None or date_el is None or not link.get("href"):
            logger.warning("ufc.com: skipping an event card without link or date")
            continue
        url = urljoin(base_url, link["href"].split("#")[0])
        ev_date = _ufc_event_date(date_el)
        if ev_date is None:
            logger.warning("ufc.com: could not read the date of %s; skipping", url)
            continue
        if url in seen:
            continue
        seen.add(url)
        loc_el = card.select_one(".c-card-event--result__location")
        parts = [t.strip() for t in loc_el.stripped_strings if t.strip(" ,")] if loc_el is not None else []
        events.append(ScheduledEvent(url=url, headline=link.get_text(" ", strip=True),
                                     event_date=ev_date.isoformat(), location=", ".join(parts) or None))
    return sorted(events, key=lambda e: e.event_date)


def _ufc_event_date(date_el) -> date | None:
    """Month/day from the displayed text (US Eastern), year from the nearest Unix timestamp."""
    for kind in ("main-card", "prelims-card", "early-card"):
        text = date_el.get(f"data-{kind}") or ""
        ts = (date_el.get(f"data-{kind}-timestamp") or "").strip()
        m = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2})", text)
        if not (m and ts.isdigit()):
            continue
        utc = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
        try:
            month = datetime.strptime(m.group(1), "%b").month
            options = [date(y, month, int(m.group(2))) for y in (utc.year - 1, utc.year, utc.year + 1)]
        except ValueError:
            continue
        return min(options, key=lambda d: abs((d - utc).days))
    return None


_SEGMENT_IDS = ("main-card", "prelims-card", "early-prelims")  # page order = bout order


def parse_ufc_event_page(html: str, event: ScheduledEvent) -> Card:
    """Bouts in listed order (bout_order 1 = main event). Red corner is fighter_1.

    ufc.com does not state the number of rounds: 5 for the main event and title bouts, else 3.
    Only names, weight class and title status are read (not odds, ranks or countries).
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    event_name = clean_event_name(re.sub(r"\s*\|\s*UFC\s*$", "", title))
    if not event_name or event_name.upper() == "UFC":
        event_name = f"UFC: {event.headline}"

    fights = [f for seg_id in _SEGMENT_IDS if (seg := soup.find(id=seg_id)) is not None
              for f in seg.select(".c-listing-fight")]
    if not fights:  # layout without segment ids
        fights = soup.select(".c-listing-fight")

    bouts = []
    for f in fights:
        names = [_corner_name(f, c) for c in ("red", "blue")]
        cls_el = f.select_one(".c-listing-fight__class-text")
        cls = cls_el.get_text(" ", strip=True) if cls_el else ""
        if not all(names):
            logger.warning("ufc.com %s: skipping bout with missing fighter name(s) %r", event.url, names)
            continue
        if (status := (f.get("data-status") or "").strip()):
            logger.info("ufc.com %s: %s vs %s has status %r", event.url, names[0], names[1], status)
        order = len(bouts) + 1
        is_title = int(bool(re.search(r"title|championship", cls, re.I)))
        bouts.append(Bout(fighter_1=names[0], fighter_2=names[1],
                          weight_class=canonical_weight_class(cls) or (cls.removesuffix(" Bout") or None),
                          bout_order=order, is_title_fight=is_title,
                          scheduled_rounds=5 if (order == 1 or is_title) else 3))
    return Card(event_name=event_name, event_date=event.event_date, source="ufc.com", bouts=bouts,
                location=event.location, event_url=event.url)


def clean_event_name(name: str) -> str:
    """'Polymarket UFC 334: A vs B' -> 'UFC 334: A vs B'; 'UFC Fight Night | A vs B' ->
    'UFC Fight Night: A vs B'. Sponsor prefixes are dropped ("Noche UFC" is kept: it is the
    event's own name)."""
    name = " ".join(name.split())
    m = re.search(r"\bUFC\b", name)
    if m and m.start() > 0 and not name[:m.start()].strip().lower().endswith("noche"):
        name = name[m.start():]
    return re.sub(r"^((?:Noche )?UFC(?: Fight Night| \d+)?)\s*[|:\-–]\s*", r"\1: ", name).strip()


def _corner_name(fight, corner: str) -> str | None:
    el = fight.select_one(f".c-listing-fight__corner-name--{corner}")
    if el is None:
        return None
    return " ".join(el.get_text(" ", strip=True).split()) or None


def card_to_json(card: Card, fetched_at: str) -> dict:
    return {
        "event_name": card.event_name, "event_date": card.event_date, "location": card.location,
        "source": card.source, "event_url": card.event_url, "fetched_at": fetched_at,
        "bouts": [{"bout_order": b.bout_order, "weight_class": b.weight_class, "fighter_1": b.fighter_1,
                   "fighter_2": b.fighter_2, "is_title_fight": b.is_title_fight,
                   "scheduled_rounds": b.scheduled_rounds} for b in card.bouts],
        "changes": card.changes,
    }


def diff_bouts(old: list[tuple[str, str]], new: list[tuple[str, str]]) -> dict | None:
    """Line-up change between two versions of a card, or None if the bouts are the same.

    Bouts are compared as unordered pairs of normalised names, so corner swaps, re-ordering and
    accent/punctuation changes are not changes. A removed and an added bout that share exactly
    one fighter are reported as a replacement ({"out", "in", "opponent"}); the rest as
    "added"/"removed" bouts ([fighter_1, fighter_2], display names as ufc.com shows them)."""
    def keyed(pairs):
        return {tuple(sorted((normalize_name(a), normalize_name(b)))): (a, b) for a, b in pairs}

    before, after = keyed(old), keyed(new)
    removed = [k for k in before if k not in after]
    added = [k for k in after if k not in before]
    replaced = []
    for r in list(removed):
        for a in added:
            common = set(r) & set(a)
            if len(common) != 1:
                continue
            (stay,) = common
            out_name = next(n for n in before[r] if normalize_name(n) != stay)
            in_name = next(n for n in after[a] if normalize_name(n) != stay)
            opponent = next(n for n in after[a] if normalize_name(n) == stay)
            replaced.append({"out": out_name, "in": in_name, "opponent": opponent})
            removed.remove(r)
            added.remove(a)
            break
    if not (removed or added or replaced):
        return None
    return {"replaced": replaced, "added": [list(after[k]) for k in added],
            "removed": [list(before[k]) for k in removed]}


def describe_change(change: dict) -> str:
    """One line of text for a `diff_bouts` result (logs and console)."""
    parts = [f"{r['in']} replaces {r['out']} (vs {r['opponent']})" for r in change.get("replaced", [])]
    parts += [f"added {a} vs {b}" for a, b in change.get("added", [])]
    parts += [f"removed {a} vs {b}" for a, b in change.get("removed", [])]
    return "; ".join(parts)


def card_filename(card: Card) -> str:
    return f"{card.event_date}_{slugify(card.event_name)}.json"


def fetch_ufc_schedule(client: HttpClient | None = None, cfg: dict | None = None,
                       today: date | None = None) -> list[Card]:
    """Fetch ufc.com/events and every upcoming event page; return cards in date order.

    Raises FetchError if ufc.com refuses the events page (403, challenge page). An event page
    that fails is logged and skipped. Requests go through src.http (global 1 req/s plus the
    configured per-host crawl delay).
    """
    cfg = cfg or load_config()
    client = client or get_client()
    today = today or datetime.now(timezone.utc).date()
    up = cfg["upcoming"]
    html = client.fetch("ufc_com", "events", up["ufc_events_url"], refresh=True)
    events = []
    for e in parse_ufc_events_page(html, up["ufc_base_url"]):
        if date.fromisoformat(e.event_date) < today:
            continue
        if any(m in e.url.lower().replace("-", " ") for m in _NON_UFC_EVENT_MARKERS):
            logger.info("ufc.com: skipping non-UFC-card event %s", e.url)
            continue
        events.append(e)
    logger.info("ufc.com: %d upcoming events", len(events))

    cards = []
    for e in events:
        slug = slugify(e.url.rstrip("/").rsplit("/", 1)[-1]) or "event"
        try:
            page = client.fetch("ufc_com_event", slug, e.url, refresh=True)
        except FetchError as err:
            logger.error("ufc.com: could not fetch %s (%s); skipping this event", e.url, err.reason)
            continue
        card = parse_ufc_event_page(page, e)
        logger.info("ufc.com: %s on %s, %d bouts", card.event_name, card.event_date, len(card.bouts))
        cards.append(card)
    return cards


def save_cards(cards: list[Card], cards_dir: Path, manual_card_file: Path, today: date,
               fetched_at: str | None = None) -> list[Path]:
    """Write one JSON per card, replace outdated files for the same event, sync the next card.

    - A ufc.com file for the same `event_url` under another name (headliner changed) is replaced.
    - Future ufc.com card files for events no longer on the schedule are removed (cancelled or
      moved); hand-made files and past cards are never touched.
    - `manual_card_file` gets the earliest card with bouts. A hand-made file there (no
      `"source": "ufc.com"`) is first moved to a timestamped backup next to it.
    - Line-up changes against the previous file for the same event (`diff_bouts`) are appended
      to the card's `changes`, stamped `detected_at = fetched_at`; earlier changes are kept.
    """
    fetched_at = fetched_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cards_dir.mkdir(parents=True, exist_ok=True)
    keep = {card_filename(c) for c in cards}
    urls = {c.event_url for c in cards}
    existing = {p: _read_json(p) for p in sorted(cards_dir.glob("*.json"))}
    previous = {d.get("event_url"): d for d in existing.values()
                if d is not None and d.get("source") == "ufc.com" and d.get("event_url")}
    for c in cards:
        old = previous.get(c.event_url)
        if old is None:
            continue
        c.changes = list(old.get("changes") or [])
        change = diff_bouts([(b["fighter_1"], b["fighter_2"]) for b in old.get("bouts") or []],
                            [(b.fighter_1, b.fighter_2) for b in c.bouts])
        if change:
            c.changes.append({"detected_at": fetched_at, **change})
            logger.warning("Card change for %s: %s", c.event_name, describe_change(change))

    for path, old in existing.items():
        if path.name in keep:
            continue
        if old is None or old.get("source") != "ufc.com":
            continue
        if old.get("event_url") in urls:
            logger.info("Replacing outdated card file %s", path.name)
            path.unlink()
        elif str(old.get("event_date", "")) >= today.isoformat():
            logger.warning("Removing %s: %s is no longer on the ufc.com schedule",
                           path.name, old.get("event_name"))
            path.unlink()

    written = []
    for c in cards:
        path = cards_dir / card_filename(c)
        _write_json(path, card_to_json(c, fetched_at))
        written.append(path)

    with_bouts = [c for c in cards if c.bouts]
    if with_bouts:
        nxt = min(with_bouts, key=lambda c: c.event_date)
        old = _read_json(manual_card_file) if manual_card_file.exists() else None
        if old is not None and old.get("source") != "ufc.com":
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup = manual_card_file.with_name(f"{manual_card_file.stem}.manual-{stamp}.json")
            os.replace(manual_card_file, backup)
            logger.warning("Moved hand-made %s to %s before syncing from ufc.com",
                           manual_card_file.name, backup.name)
        _write_json(manual_card_file, card_to_json(nxt, fetched_at))
        logger.info("Synced %s: %s on %s", manual_card_file.name, nxt.event_name, nxt.event_date)
    return written


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Cannot read %s (%s); leaving it", path, e)
        return None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_schedule(client: HttpClient | None = None, today: date | None = None) -> list[Card]:
    """Fetch ufc.com and save the cards (see `save_cards`). Raises FetchError if refused."""
    cfg = load_config()
    today = today or datetime.now(timezone.utc).date()
    cards = fetch_ufc_schedule(client, cfg, today)
    if not cards:
        raise UpcomingCardError("ufc.com listed no upcoming UFC events")
    save_cards(cards, resolve_path(cfg["upcoming"]["cards_dir"]),
               resolve_path(cfg["upcoming"]["manual_card_file"]), today)
    return cards


def load_scheduled_cards(today: date | None = None) -> list[tuple[Path, Card]]:
    """Every card file in cards_dir dated today or later, in date order."""
    cfg = load_config()
    today = today or datetime.now(timezone.utc).date()
    out = []
    for path in sorted(resolve_path(cfg["upcoming"]["cards_dir"]).glob("*.json")):
        if path.name[:10] < today.isoformat():  # <date>_<slug>.json: past card, skip quietly
            continue
        card = load_manual_card(path, today)
        if card is not None:
            out.append((path, card))
    return sorted(out, key=lambda pc: (pc[1].event_date, pc[0].name))


# --------------------------------------------------------------------------- ESPN

def parse_espn_scoreboard(data: dict, today: date) -> Card | None:
    """Pick the earliest not-yet-completed UFC card on/after `today` from a scoreboard response.

    Structure used: events[].{name, date, status.type.completed, competitions[]}, and per
    competition: competitors[].{order, athlete.displayName|fullName}, type.abbreviation|text
    (weight class), status.type.completed.
    """
    candidates = []
    for ev in data.get("events") or []:
        name = ev.get("name") or ev.get("shortName") or ""
        if any(m in name.lower() for m in _NON_UFC_EVENT_MARKERS):
            logger.info("ESPN: skipping non-UFC-card event %r", name)
            continue
        ev_date = _espn_date(ev.get("date"))
        if ev_date is None or ev_date < today:
            continue
        if ((ev.get("status") or {}).get("type") or {}).get("completed"):
            continue
        candidates.append((ev_date, name, ev))
    if not candidates:
        return None

    ev_date, name, ev = min(candidates, key=lambda c: c[0])
    bouts = []
    for comp in ev.get("competitions") or []:
        comps = sorted(comp.get("competitors") or [], key=lambda c: c.get("order", 0))
        names = [_athlete_name(c) for c in comps]
        if len(names) != 2 or not all(names):
            logger.warning("ESPN: skipping competition %s with competitors %r", comp.get("id"), names)
            continue
        wc_raw = (comp.get("type") or {}).get("abbreviation") or (comp.get("type") or {}).get("text")
        bouts.append(Bout(fighter_1=names[0], fighter_2=names[1],
                          weight_class=canonical_weight_class(wc_raw) or wc_raw))
    # ESPN order is not guaranteed to be main event first; number as listed.
    for i, b in enumerate(bouts, start=1):
        b.bout_order = i
    return Card(event_name=name, event_date=ev_date.isoformat(), source="espn", bouts=bouts)


def fetch_espn_card(client: HttpClient, cfg: dict, today: date) -> Card | None:
    up = cfg["upcoming"]
    end = today + timedelta(days=up["lookahead_days"])
    params = {"dates": f"{today:%Y%m%d}-{end:%Y%m%d}"}
    raw = client.get(up["espn_scoreboard_url"], params=params)
    cache_dir = resolve_path(up["espn_cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"scoreboard_{today:%Y%m%d}.json").write_bytes(raw)
    return parse_espn_scoreboard(json.loads(raw), today)


def _espn_date(value) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _athlete_name(competitor: dict) -> str | None:
    a = competitor.get("athlete") or {}
    return a.get("displayName") or a.get("fullName") or competitor.get("displayName")


# --------------------------------------------------------------------------- matching

def load_aliases(path: Path | None = None, known_ids: set[str] | None = None) -> dict[str, str]:
    """Committed, hand-verified card-name -> fighter_id fixes (`upcoming_aliases.csv`:
    name, fighter_id, note), keyed by normalised name. Entries whose fighter_id is not in
    `known_ids` are ignored with a warning."""
    if path is None:
        path = resolve_path(load_config()["upcoming"]["aliases_file"])
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, comment="#").fillna("")
    aliases = {}
    for r in df.itertuples(index=False):
        if not r.name.strip() or not r.fighter_id.strip():
            continue
        if known_ids is not None and r.fighter_id.strip() not in known_ids:
            logger.warning("%s: fighter_id %s for %r is not in fighters.csv; ignoring", path.name,
                           r.fighter_id, r.name)
            continue
        aliases[normalize_name(r.name)] = r.fighter_id.strip()
    return aliases


def match_card(card: Card, matcher: FighterMatcher, aliases: dict[str, str] | None = None) -> Card:
    """Fill fighter ids that weren't given manually: alias file first, then `matcher`.
    Unmatched names keep fighter_id None."""
    on_date = date.fromisoformat(card.event_date)
    aliases = aliases or {}
    for b in card.bouts:
        for side in ("1", "2"):
            if getattr(b, f"fighter_{side}_id"):
                continue
            name = getattr(b, f"fighter_{side}")
            if (alias := aliases.get(normalize_name(name))):
                setattr(b, f"fighter_{side}_id", alias)
                setattr(b, f"fighter_{side}_match", "alias")
                continue
            m = matcher.match(name, weight_class=b.weight_class, on_date=on_date, fuzzy=True)
            setattr(b, f"fighter_{side}_id", m.fighter_id)
            detail = m.method
            if m.method == "ambiguous":
                detail += f" (candidates: {', '.join(m.candidates)})"
            elif not m.ok and m.candidates:
                detail += f" (closest: {m.candidates[0]}, score {m.score:.2f})"
            setattr(b, f"fighter_{side}_match", detail)
            if m.method == "fuzzy":
                logger.info("Fuzzy match %r -> %s (score %.2f)", name, m.fighter_id, m.score)
            elif not m.ok:
                logger.warning("No fighter_id for %r (%s); treated as possible debut", name, m.method)
    return card


def get_upcoming_card(fighters: pd.DataFrame | None = None, client: HttpClient | None = None,
                      today: date | None = None, card_path: Path | None = None) -> Card:
    """`card_path` or a hand-made upcoming_card.json wins; else ufc.com; else ESPN.

    An upcoming_card.json written by the ufc.com sync is refreshed from ufc.com, and used
    as-is only when ufc.com cannot be reached."""
    cfg = load_config()
    today = today or datetime.now(timezone.utc).date()
    card = load_manual_card(card_path or resolve_path(cfg["upcoming"]["manual_card_file"]), today)
    synced = card if card is not None and card.source == "ufc.com" and card_path is None else None
    errors = []
    if card is None or synced is not None:
        try:
            card = next((c for c in update_schedule(client, today) if c.bouts), None)
        except (FetchError, UpcomingCardError) as e:
            errors.append(f"ufc.com: {getattr(e, 'reason', e)}")
            if synced is not None:
                logger.warning("ufc.com refresh failed (%s); using the last synced card", errors[-1])
            card = synced
    if card is None:
        try:
            card = fetch_espn_card(client or get_client(), cfg, today)
        except FetchError as e:
            errors.append(f"ESPN: {e.reason}")
            raise UpcomingCardError(
                f"No card source available ({'; '.join(errors)}). Create "
                f"{cfg['upcoming']['manual_card_file']} (see tests/fixtures/upcoming_card.example.json).") from e
    if card is None or not card.bouts:
        raise UpcomingCardError("No upcoming UFC card found; create the manual card file.")
    logger.info("Upcoming card from %s: %s on %s, %d bouts", card.source, card.event_name,
                card.event_date, len(card.bouts))

    if fighters is None:
        fighters = pd.read_csv(resolve_path(cfg["paths"]["raw_dir"]) / "fighters.csv", dtype=str)
    return match_card(card, FighterMatcher.from_fighters(fighters), load_aliases(known_ids=set(fighters.fighter_id)))
