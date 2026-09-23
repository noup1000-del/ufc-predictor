"""Get the next UFC card: manual JSON override first, else ESPN's scoreboard API.

Fighter names are mapped to fighter_id with `src.matching`; unmatched names stay
visible (fighter_id None) rather than being guessed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.http import FetchError, HttpClient, get_client
from src.matching import FighterMatcher, canonical_weight_class

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


@dataclass
class Card:
    event_name: str
    event_date: str  # YYYY-MM-DD
    source: str      # "manual" | "espn"
    bouts: list[Bout] = field(default_factory=list)

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

def load_manual_card(path: Path, today: date) -> Card | None:
    """Read the manual card file; ignored (with a warning) if its date has passed."""
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
                 fighter_2_match="manual" if b.get("fighter_2_id") else None)
            for i, b in enumerate(data["bouts"])
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise UpcomingCardError(f"invalid manual card file {path}: {e}") from e
    if event_date < today:
        logger.warning("Manual card %s is dated %s (in the past); ignoring it", path, event_date)
        return None
    return Card(event_name=data.get("event_name", "Manual card"), event_date=event_date.isoformat(),
                source="manual", bouts=bouts)


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

def match_card(card: Card, matcher: FighterMatcher) -> Card:
    """Fill fighter ids that weren't given manually. Unmatched names keep fighter_id None."""
    on_date = date.fromisoformat(card.event_date)
    for b in card.bouts:
        for side in ("1", "2"):
            if getattr(b, f"fighter_{side}_id"):
                continue
            name = getattr(b, f"fighter_{side}")
            m = matcher.match(name, weight_class=b.weight_class, on_date=on_date, fuzzy=True)
            setattr(b, f"fighter_{side}_id", m.fighter_id)
            detail = m.method
            if not m.ok and m.candidates:
                detail += f" (candidates: {', '.join(m.candidates)})"
            setattr(b, f"fighter_{side}_match", detail)
            if m.method == "fuzzy":
                logger.info("Fuzzy match %r -> %s (score %.2f)", name, m.fighter_id, m.score)
            elif not m.ok:
                logger.warning("No fighter_id for %r (%s); treated as possible debut", name, m.method)
    return card


def get_upcoming_card(fighters: pd.DataFrame | None = None, client: HttpClient | None = None,
                      today: date | None = None) -> Card:
    cfg = load_config()
    today = today or datetime.now(timezone.utc).date()
    card = load_manual_card(resolve_path(cfg["upcoming"]["manual_card_file"]), today)
    if card is None:
        try:
            card = fetch_espn_card(client or get_client(), cfg, today)
        except FetchError as e:
            raise UpcomingCardError(
                f"ESPN request failed ({e.reason}). Create {cfg['upcoming']['manual_card_file']} "
                "(see tests/fixtures/upcoming_card.example.json).") from e
    if card is None or not card.bouts:
        raise UpcomingCardError("No upcoming UFC card found; create the manual card file.")
    logger.info("Upcoming card from %s: %s on %s, %d bouts", card.source, card.event_name,
                card.event_date, len(card.bouts))

    if fighters is None:
        fighters = pd.read_csv(resolve_path(cfg["paths"]["raw_dir"]) / "fighters.csv", dtype=str)
    return match_card(card, FighterMatcher.from_fighters(fighters))
