"""Fighter-name -> fighter_id matching and weight-class helpers.

Names are only ever joined here (used by ingest for the name-only archive and by
upcoming-card matching). Everything downstream joins on fighter_id.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# Checked in order: "light heavyweight" must come before "heavyweight".
_DIVISIONS = [
    ("light heavyweight", "Light Heavyweight", 205),
    ("super heavyweight", "Super Heavyweight", None),
    ("heavyweight", "Heavyweight", 265),
    ("middleweight", "Middleweight", 185),
    ("welterweight", "Welterweight", 170),
    ("lightweight", "Lightweight", 155),
    ("featherweight", "Featherweight", 145),
    ("bantamweight", "Bantamweight", 135),
    ("flyweight", "Flyweight", 125),
    ("strawweight", "Strawweight", 115),
    ("catch weight", "Catch Weight", None),
    ("catchweight", "Catch Weight", None),
    ("open weight", "Open Weight", None),
    ("openweight", "Open Weight", None),
]

MIN_AGE, MAX_AGE = 18, 50
WEIGHT_MARGIN_LB = 15
FUZZY_MIN_SCORE = 0.92
FUZZY_MIN_MARGIN = 0.05


def canonical_weight_class(raw: str | None) -> str | None:
    """'UFC Women's Strawweight Title Bout' -> "Women's Strawweight"; None if no division named."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = re.sub(r"\s+", " ", str(raw).lower().replace("’", "'"))
    for key, name, _ in _DIVISIONS:
        if key in text:
            return f"Women's {name}" if "women" in text else name
    return None


def weight_limit_lb(weight_class: str | None) -> int | None:
    wc = canonical_weight_class(weight_class)
    if wc is None:
        return None
    base = wc.removeprefix("Women's ")
    return next((lim for _, name, lim in _DIVISIONS if name == base), None)


# Letters that NFKD does not decompose into base letter + accent.
_TRANSLIT = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss",
                           "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i", "þ": "th"})


def normalize_name(name: str) -> str:
    """Accent-stripped, lowercase, punctuation-free, single-spaced."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name.translate(_TRANSLIT))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[.'’`\"]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def _compact(norm: str) -> str:
    return norm.replace(" ", "")


def _similarity(a: str, b: str) -> float:
    sorted_a, sorted_b = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
    return max(
        SequenceMatcher(None, a, b).ratio(),
        SequenceMatcher(None, sorted_a, sorted_b).ratio(),
    )


@dataclass(frozen=True)
class FighterInfo:
    weight_lb: float | None = None
    dob: date | None = None


@dataclass(frozen=True)
class Match:
    fighter_id: str | None
    method: str  # exact | exact_disambiguated | compact | fuzzy | ambiguous | unmatched
    score: float = 0.0
    candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.fighter_id is not None


class FighterMatcher:
    def __init__(self, names: Iterable[tuple[str, str]], info: dict[str, FighterInfo] | None = None):
        """`names`: (fighter_id, name) pairs; a fighter may appear under several names."""
        self.info = info or {}
        self._by_norm: dict[str, set[str]] = {}
        self._by_compact: dict[str, set[str]] = {}
        for fid, name in names:
            norm = normalize_name(name)
            if not norm or not fid:
                continue
            self._by_norm.setdefault(norm, set()).add(fid)
            self._by_compact.setdefault(_compact(norm), set()).add(fid)

    @classmethod
    def from_fighters(cls, fighters: pd.DataFrame) -> "FighterMatcher":
        """Build from a fighters.csv-shaped frame (fighter_id, name, weight_lb, dob)."""
        return cls(zip(fighters.fighter_id, fighters.name), fighter_info(fighters))

    def match(self, name: str, weight_class: str | None = None,
              on_date: date | None = None, fuzzy: bool = False) -> Match:
        norm = normalize_name(name)
        if not norm:
            return Match(None, "unmatched")

        ids = self._by_norm.get(norm)
        if ids:
            return self._pick(ids, weight_class, on_date, "exact")
        if not fuzzy:
            return Match(None, "unmatched")

        ids = self._by_compact.get(_compact(norm))
        if ids:
            return self._pick(ids, weight_class, on_date, "compact")

        scored = sorted(((_similarity(norm, cand), cand) for cand in self._by_norm), reverse=True)
        if not scored:
            return Match(None, "unmatched")
        best_score, best = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= FUZZY_MIN_SCORE and best_score - runner_up >= FUZZY_MIN_MARGIN:
            m = self._pick(self._by_norm[best], weight_class, on_date, "fuzzy")
            return Match(m.fighter_id, m.method, best_score, m.candidates)
        return Match(None, "unmatched", best_score, (best,))

    def _pick(self, ids: set[str], weight_class, on_date, method: str) -> Match:
        cands = tuple(sorted(ids))
        if len(cands) == 1:
            return Match(cands[0], method, 1.0, cands)
        chosen = self.disambiguate(cands, weight_class, on_date)
        if chosen:
            return Match(chosen, f"{method}_disambiguated" if method == "exact" else method, 1.0, cands)
        return Match(None, "ambiguous", 0.0, cands)

    def disambiguate(self, ids: Iterable[str], weight_class: str | None,
                     on_date: date | None) -> str | None:
        """Pick one of several same-name fighters, or None if the evidence isn't clear."""
        remaining = []
        for fid in ids:
            dob = self.info.get(fid, FighterInfo()).dob
            if dob and on_date:
                age = (on_date - dob).days / 365.25
                if not MIN_AGE <= age <= MAX_AGE:
                    continue
            remaining.append(fid)
        if len(remaining) == 1:
            return remaining[0]
        if len(remaining) < 2:
            return None

        limit = weight_limit_lb(weight_class)
        weights = [self.info.get(fid, FighterInfo()).weight_lb for fid in remaining]
        if limit is None or any(w is None for w in weights):
            return None
        ranked = sorted(zip((abs(w - limit) for w in weights), remaining))
        if ranked[0][0] + WEIGHT_MARGIN_LB <= ranked[1][0]:
            return ranked[0][1]
        return None


def fighter_info(fighters: pd.DataFrame) -> dict[str, FighterInfo]:
    """fighter_id -> FighterInfo from a frame with fighter_id, weight_lb, dob columns."""
    return {
        r.fighter_id: FighterInfo(_num_or_none(r.weight_lb), _date_or_none(r.dob))
        for r in fighters.itertuples(index=False)
    }


def _num_or_none(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def _date_or_none(v) -> date | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        return None
