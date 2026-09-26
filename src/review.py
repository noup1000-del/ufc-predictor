"""Stage 7: post-event review ("lessons learned") from tracked results.

For every event in outputs/tracking/results_log.csv this writes
outputs/reviews/<event_date>_<event_slug>.json with
  - a scorecard per bout: pick, probability, actual winner, method/round, how surprising a
    miss was, what the model relied on (its saved key factors, not a post-hoc story), and
    whether the bout involved a debutant or a late line-up change;
  - event lessons: rule-based statements (hits vs the model's own expectation, biggest
    surprises, debut/late-change/finish breakdowns). One card is never treated as evidence.
and outputs/reviews/evidence.json: the evidence board over ALL reviewed fights. For each
segment (confidence band, debut, late change, method, ...) it compares the hit rate with the
model's own expected hit rate (mean pick probability). A segment is flagged only when it has
at least MIN_EVIDENCE_FIGHTS fights AND the 95% Wilson interval of the hit rate excludes the
expectation; otherwise it is "collecting evidence". Model changes are proposed only from
flagged segments, and still go through the backtest/promotion gate in train.py.

Joins: log rows -> fights by fight_id; the prediction row used for a fight is found by
fighter IDs within that prediction file (track.match_predictions), never by names. Card
changes (ufc.com names) are looked up by normalised name inside the same card only.

Usage: python -m src.review
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import load_processed
from src.config import load_config, resolve_path
from src.matching import normalize_name
from src.track import load_predictions, match_predictions
from src.train import wilson_interval
from src.upcoming import slugify

logger = logging.getLogger(__name__)

MIN_EVIDENCE_FIGHTS = 30          # below this a segment is "collecting evidence", whatever it shows
COIN_FLIP, CLEAR_FAVOURITE = 0.55, 0.65
SURPRISE = [(COIN_FLIP, "coin flip"), (CLEAR_FAVOURITE, "lean"), (1.01, "clear favourite")]
METHOD_LABELS = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision", "dq": "DQ",
                 "other": "Other"}


def surprise_level(p_pick: float) -> str:
    """How surprising it is when the pick loses: <55% coin flip, <65% lean, else clear favourite."""
    return next(label for limit, label in SURPRISE if p_pick < limit)


def parse_key_factors(text, fighter_1: str, fighter_2: str) -> dict:
    """'A: f1; f2 | B: none' -> {fighter_1: [...], fighter_2: [...]} (predict.key_factors_text)."""
    out = {"fighter_1": [], "fighter_2": []}
    if not isinstance(text, str) or not text:
        return out
    left, sep, right = text.partition(" | ")
    for side, name, part in (("fighter_1", fighter_1, left), ("fighter_2", fighter_2, right)):
        prefix = f"{name}: "
        if part.startswith(prefix):
            body = part[len(prefix):]
            out[side] = [] if body == "none" else [_WEIGHT.sub("", x).strip() for x in body.split("; ") if x.strip()]
    return out


# "Age 22.0 vs 39.4 (0.35)" -> drop the trailing log-odds weight; it means nothing to readers.
_WEIGHT = re.compile(r"\s*\(\d+(?:\.\d+)?\)$")


def _late_changes(cards_dir: Path, event_date: str) -> tuple[set[str], set[frozenset], dict]:
    """Normalised names that came in as replacements, pairs of added bouts, and the detection
    time per name/pair, from the ufc.com card file(s) of that date."""
    names, pairs, when = set(), set(), {}
    for path in sorted(cards_dir.glob(f"{event_date}_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Cannot read %s (%s); late changes for it unknown", path.name, e)
            continue
        for c in data.get("changes") or []:
            for r in c.get("replaced", []):
                n = normalize_name(r["in"])
                names.add(n)
                when.setdefault(n, c.get("detected_at"))
            for a, b in c.get("added", []):
                key = frozenset((normalize_name(a), normalize_name(b)))
                pairs.add(key)
                when.setdefault(key, c.get("detected_at"))
    return names, pairs, when


def bout_records(log: pd.DataFrame, fights: pd.DataFrame, preds: pd.DataFrame, cards_dir: Path) -> pd.DataFrame:
    """One row per logged fight with everything the review needs."""
    f = fights.set_index("fight_id")
    used = match_predictions(preds[preds["prediction_file"].isin(set(log["prediction_file"]))], fights)
    used = used[used["status"] == "matched"].set_index(["prediction_file", "fight_id"])
    late_cache: dict[str, tuple] = {}
    rows = []
    for r in log.itertuples(index=False):
        fight = f.loc[r.fight_id]
        try:
            p = used.loc[(r.prediction_file, r.fight_id)]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[-1]
        except KeyError:
            logger.warning("Prediction row for %s (%s) not found in %s", r.fight_id, r.event_name, r.prediction_file)
            p = pd.Series(dtype=object)
        if r.event_date not in late_cache:
            late_cache[r.event_date] = _late_changes(cards_dir, r.event_date)
        late_names, late_pairs, late_when = late_cache[r.event_date]
        n1, n2 = normalize_name(r.fighter_1), normalize_name(r.fighter_2)
        late_keys = [k for k in (n1, n2, frozenset((n1, n2))) if k in late_names or k in late_pairs]
        correct = None if pd.isna(r.correct) or str(r.correct) == "" else str(r.correct).lower() == "true"
        p_pick = float(r.p_predicted_winner)
        drivers = parse_key_factors(p.get("key_factors"), r.fighter_1, r.fighter_2)
        pick_side = "fighter_1" if r.predicted_winner == r.fighter_1 else "fighter_2"
        other_side = "fighter_2" if pick_side == "fighter_1" else "fighter_1"
        rows.append({
            "event_date": r.event_date, "event_name": r.event_name, "fight_id": r.fight_id,
            "bout_order": _int(p.get("bout_order")), "weight_class": _str(fight.get("weight_class")),
            "fighter_1": r.fighter_1, "fighter_2": r.fighter_2,
            "pick": r.predicted_winner, "p_pick": p_pick, "actual_winner": r.actual_winner,
            "correct": correct, "result": _str(fight.get("result")),
            "method_group": _str(fight.get("method_group")), "method": _str(fight.get("method")),
            "end_round": _int(fight.get("end_round")), "end_time_sec": _int(fight.get("end_time_sec")),
            "scheduled_rounds": _int(fight.get("scheduled_rounds")),
            "surprise": surprise_level(p_pick) if correct is False else None,
            "debut": bool(_int(p.get("fighter_1_debut")) or _int(p.get("fighter_2_debut"))),
            "debutants": [n for n, d in ((r.fighter_1, p.get("fighter_1_debut")), (r.fighter_2, p.get("fighter_2_debut")))
                          if _int(d)],
            "late_change": bool(late_keys),
            "late_change_detected_at": min((late_when[k] for k in late_keys if late_when.get(k)), default=None),
            "pick_factors": drivers[pick_side], "other_factors": drivers[other_side],
            "model_version": r.model_version, "prediction_file": r.prediction_file,
        })
    return pd.DataFrame(rows)


def _int(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else int(f)


def _str(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v) else str(v)


def _scored(b: pd.DataFrame) -> pd.DataFrame:
    return b[b["correct"].notna()].assign(y=lambda d: d["correct"].astype(float))


def _summary(s: pd.DataFrame) -> dict:
    if s.empty:
        return {"fights": 0}
    y, p = s["y"].to_numpy(), s["p_pick"].to_numpy(dtype=float)
    return {"fights": int(len(s)), "correct": int(y.sum()), "accuracy": float(y.mean()),
            "expected_correct": float(p.sum()), "expected_accuracy": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))))}


def event_lessons(b: pd.DataFrame) -> list[str]:
    """Rule-based lessons for one event. Statements only; no model change follows from one card."""
    s = _scored(b)
    out = []
    if s.empty:
        return ["No scored fights (draws/no contests only)."]
    m = _summary(s)
    sd = math.sqrt(float(np.sum(s["p_pick"] * (1 - s["p_pick"]))))
    diff = m["correct"] - m["expected_correct"]
    verdict = ("in line with" if abs(diff) <= 1.5 * sd else "better than" if diff > 0 else "worse than")
    out.append(f"{m['correct']} of {m['fights']} picks correct ({m['accuracy']:.0%}). From its own probabilities "
               f"the model expected about {m['expected_correct']:.1f} correct, so this card went {verdict} "
               f"expected (normal night-to-night spread is about ±{sd:.1f}).")
    misses = s[s["y"] == 0].sort_values("p_pick", ascending=False)
    for r in misses[misses["p_pick"] >= CLEAR_FAVOURITE].itertuples(index=False):
        how = _how(r)
        why = ("; ".join(r.pick_factors[:3]) + f" (values: {r.fighter_1} vs {r.fighter_2})"
               if r.pick_factors else "no single strong factor")
        out.append(f"Upset: {r.actual_winner} beat {r.pick} ({r.p_pick:.0%} for {r.pick}){how}. "
                   f"The model favoured {r.pick} mainly on: {why}.")
    flips = misses[misses["p_pick"] < COIN_FLIP]
    if len(flips):
        out.append(f"{len(flips)} miss(es) were near coin flips (under {COIN_FLIP:.0%}): "
                   + ", ".join(f"{r.actual_winner} over {r.pick}" for r in flips.itertuples(index=False))
                   + ". These say little about the model.")
    for label, mask in (("involving a UFC debutant", s["debut"]), ("with a late line-up change", s["late_change"])):
        seg = s[mask]
        if len(seg):
            out.append(f"Bouts {label}: {int(seg['y'].sum())} of {len(seg)} correct "
                       f"(model expected {seg['p_pick'].sum():.1f}).")
    fin = s[s["method_group"].isin(["ko_tko", "submission"])]
    if len(fin):
        out.append(f"Finishes: {int(fin['y'].sum())} of {len(fin)} picked correctly; decisions: "
                   f"{int(s.loc[s['method_group'] == 'decision', 'y'].sum())} of "
                   f"{int((s['method_group'] == 'decision').sum())}.")
    nc = b[b["correct"].isna()]
    if len(nc):
        out.append(f"{len(nc)} bout(s) ended in a draw or no contest and are not scored.")
    out.append("One card is not evidence: patterns are judged on the evidence board across all tracked fights.")
    return out


def _how(r) -> str:
    method = METHOD_LABELS.get(r.method_group or "", r.method_group or "")
    if not method:
        return ""
    if r.method_group == "decision" or not r.end_round:
        return f" by {method.lower()}"
    return f" by {method} in round {r.end_round}"


SEGMENTS = {
    "Pick confidence": lambda s: pd.cut(s["p_pick"], [0, COIN_FLIP, CLEAR_FAVOURITE, 1.0001], right=False,
                                        labels=["under 55%", "55-65%", "65% and up"]).astype(str),
    "UFC debutant in the bout": lambda s: s["debut"].map({True: "yes", False: "no"}),
    "Late line-up change": lambda s: s["late_change"].map({True: "yes", False: "no"}),
    "How the fight ended": lambda s: s["method_group"].map(METHOD_LABELS).fillna("Unknown"),
    "Scheduled rounds": lambda s: s["scheduled_rounds"].map({5: "5 rounds", 3: "3 rounds"}).fillna("Unknown"),
    "Division": lambda s: s["weight_class"].fillna("").str.startswith("Women").map({True: "Women", False: "Men"}),
}


def evidence_board(b: pd.DataFrame, min_fights: int = MIN_EVIDENCE_FIGHTS) -> dict:
    """Hit rate vs the model's own expected hit rate per segment, over all reviewed fights."""
    s = _scored(b)
    board = {"overall": _summary(s), "min_fights": min_fights, "segments": []}
    if s.empty:
        return board
    for name, fn in SEGMENTS.items():
        for value, g in s.groupby(fn(s)):
            n, k = len(g), int(g["y"].sum())
            lo, hi = (float(x) for x in wilson_interval(k, n))
            expected = float(g["p_pick"].mean())
            if n < min_fights:
                verdict, status = f"collecting evidence ({n}/{min_fights} fights)", "collecting"
            elif expected > hi:
                verdict, status = "evidence: model is over-confident here", "overconfident"
            elif expected < lo:
                verdict, status = "evidence: model is under-confident here", "underconfident"
            else:
                verdict, status = "consistent with the model's own expectation", "consistent"
            board["segments"].append({"segment": name, "value": str(value), "fights": n, "correct": k,
                                      "accuracy": k / n, "expected_accuracy": expected, "ci_low": lo,
                                      "ci_high": hi, "status": status, "verdict": verdict})
    return board


def build_reviews(b: pd.DataFrame, generated: str) -> tuple[list[dict], dict]:
    reviews = []
    for (event_date, event_name), g in b.groupby(["event_date", "event_name"], sort=True):
        g = g.sort_values(["bout_order", "fight_id"], na_position="last")
        reviews.append({
            "event_date": event_date, "event_name": event_name, "slug": slugify(event_name),
            "generated_at": generated, "summary": _summary(_scored(g)),
            "lessons": event_lessons(g),
            "bouts": json.loads(g.drop(columns=["event_date", "event_name"]).to_json(orient="records")),
        })
    board = evidence_board(b)
    board.update(generated_at=generated, events=len(reviews))
    return reviews, board


def load_reviews(reviews_dir: Path | None = None) -> tuple[list[dict], dict | None]:
    """Saved reviews (newest event first) and the evidence board, for the dashboard."""
    reviews_dir = reviews_dir or resolve_path(load_config()["paths"]["reviews_dir"])
    reviews = []
    for path in sorted(reviews_dir.glob("20*.json"), reverse=True):
        try:
            reviews.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Cannot read review %s (%s); skipping", path.name, e)
    board_path = reviews_dir / "evidence.json"
    board = json.loads(board_path.read_text(encoding="utf-8")) if board_path.exists() else None
    return reviews, board


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def run_review(generated: str | None = None) -> dict:
    cfg = load_config()
    generated = generated or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    log_path = resolve_path(cfg["paths"]["tracking_dir"]) / "results_log.csv"
    if not log_path.exists():
        logger.info("No results log yet; nothing to review")
        return {"events": 0, "fights": 0}
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False, na_values=[""])
    if log.empty:
        logger.info("Results log is empty; nothing to review")
        return {"events": 0, "fights": 0}
    fights = load_processed()["fights"].astype({"fight_id": str, "fighter_1_id": str, "fighter_2_id": str,
                                                "winner_id": "string", "result": str})
    preds = load_predictions(resolve_path(cfg["paths"]["predictions_dir"]))
    b = bout_records(log, fights, preds, resolve_path(cfg["upcoming"]["cards_dir"]))
    reviews, board = build_reviews(b, generated)

    out_dir = resolve_path(cfg["paths"]["reviews_dir"])
    for r in reviews:
        _write_json(out_dir / f"{r['event_date']}_{r['slug']}.json", r)
        s = r["summary"]
        if s.get("fights"):
            logger.info("Review %s %s: %d/%d correct (expected %.1f), Brier %.3f", r["event_date"], r["event_name"],
                        s["correct"], s["fights"], s["expected_correct"], s["brier"])
    _write_json(out_dir / "evidence.json", board)
    flagged = [g for g in board["segments"] if g["status"] in ("overconfident", "underconfident")]
    for g in flagged:
        logger.warning("Evidence board: %s = %s: %s (%d fights, hit rate %.0f%%, expected %.0f%%)", g["segment"],
                       g["value"], g["verdict"], g["fights"], 100 * g["accuracy"], 100 * g["expected_accuracy"])
    return {"events": len(reviews), "fights": int(board["overall"].get("fights", 0)), "flagged": len(flagged)}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_review()
    except Exception:
        logger.exception("Review failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
