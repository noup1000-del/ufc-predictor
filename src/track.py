"""Stage 6: match saved predictions to completed fights and keep a running log.

- Reads every outputs/predictions/*.csv and matches each bout to a processed fight by
  fighter IDs and date: same unordered fighter pair (card order may differ from the
  source's), event_date within +-1 day (ESPN dates are UTC). If one fighter was
  unmatched at prediction time (debut), the other fighter's fight on that date is used.
- Appends new rows to outputs/tracking/results_log.csv, keyed by fight_id: a fight is
  logged once and existing rows are never rewritten (idempotent).
- Predictions made after the event (predicted_at later than event_date + 1 day) are not
  counted.
- Draws/NCs are logged with `correct` empty and excluded from metrics.

Usage: python -m src.track
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import load_processed
from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

LOG_COLUMNS = ["event_date", "fighter_1", "fighter_2", "predicted_winner", "actual_winner", "correct",
               "p_predicted_winner", "model_version", "tracked_at"]
# Appended after the requested columns: fight_id is the idempotency key.
EXTRA_COLUMNS = ["fight_id", "event_name", "prediction_file"]
DATE_TOLERANCE_DAYS = 1
RECENT_EVENTS = 5


def load_predictions(pred_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(pred_dir.glob("*.csv")):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.error("Cannot read %s: %s", path.name, e)
            continue
        frames.append(df.assign(prediction_file=path.name))
    if not frames:
        return pd.DataFrame(columns=["event_date", "fighter_1_id", "fighter_2_id", "prediction_file"])
    return pd.concat(frames, ignore_index=True)


def match_predictions(preds: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Return one row per prediction with fight_id (or None) and a status."""
    f = fights[["fight_id", "event_date", "fighter_1_id", "fighter_2_id", "winner_id", "result"]].copy()
    f["event_date"] = pd.to_datetime(f["event_date"])
    by_fighter = pd.concat([f.rename(columns={"fighter_1_id": "fid"}), f.rename(columns={"fighter_2_id": "fid"})])
    by_fighter = by_fighter[["fid", "fight_id", "event_date"]]
    fights_by_id = f.set_index("fight_id")

    out = []
    for p in preds.itertuples(index=False):
        date = pd.Timestamp(p.event_date)
        ids = [i for i in (p.fighter_1_id, p.fighter_2_id) if isinstance(i, str) and i]
        rec = {"fight_id": None, "status": "pending"}
        if not ids:
            rec["status"] = "unmatchable (no fighter ids)"
        else:
            near = by_fighter[(by_fighter.event_date - date).abs() <= pd.Timedelta(days=DATE_TOLERANCE_DAYS)]
            cand = set(near.loc[near.fid == ids[0], "fight_id"])
            for other in ids[1:]:
                cand &= set(near.loc[near.fid == other, "fight_id"])
            if len(cand) == 1:
                fid = cand.pop()
                fight = fights_by_id.loc[fid]
                pair = {fight.fighter_1_id, fight.fighter_2_id}
                if set(ids) <= pair:
                    rec = {"fight_id": fid, "status": "matched"}
                else:
                    rec["status"] = "fighter mismatch"
            elif len(cand) > 1:
                rec["status"] = "ambiguous"
        out.append(rec)
    return pd.concat([preds.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def build_log_rows(matched: pd.DataFrame, fights: pd.DataFrame, tracked_at: str) -> pd.DataFrame:
    m = matched[matched["status"] == "matched"].copy()
    if m.empty:
        return pd.DataFrame(columns=LOG_COLUMNS + EXTRA_COLUMNS)
    # Only predictions made before the fight count.
    predicted = pd.to_datetime(m["predicted_at"], utc=True).dt.tz_localize(None)
    late = predicted > pd.to_datetime(m["event_date"]) + pd.Timedelta(days=DATE_TOLERANCE_DAYS + 1)
    for r in m[late].itertuples(index=False):
        logger.warning("Ignoring prediction made after the event: %s vs %s (%s, predicted %s)",
                       r.fighter_1, r.fighter_2, r.event_date, r.predicted_at)
    m = m[~late]
    # If several prediction files cover the same fight, keep the latest pre-fight prediction.
    m = m.sort_values("predicted_at").drop_duplicates("fight_id", keep="last")

    f = fights.set_index("fight_id")
    rows = []
    for r in m.itertuples(index=False):
        fight = f.loc[r.fight_id]
        result = str(fight["result"])
        if result == "win":
            actual = _winning_side(fight["winner_id"], r.fighter_1_id, r.fighter_2_id, r.fighter_1, r.fighter_2)
            correct = actual == r.predicted_winner
        else:
            actual, correct = result, None
        rows.append({
            "event_date": r.event_date, "fighter_1": r.fighter_1, "fighter_2": r.fighter_2,
            "predicted_winner": r.predicted_winner, "actual_winner": actual, "correct": correct,
            "p_predicted_winner": float(r.confidence), "model_version": r.model_version,
            "tracked_at": tracked_at, "fight_id": r.fight_id,
            "event_name": getattr(r, "event_name", None), "prediction_file": r.prediction_file,
        })
    return pd.DataFrame(rows, columns=LOG_COLUMNS + EXTRA_COLUMNS)


def _has(v) -> bool:
    return isinstance(v, str) and bool(v)


def _winning_side(winner_id, id_1, id_2, name_1, name_2) -> str:
    """Name (as on the prediction) of the side that won. One id may be unknown (debut)."""
    if _has(id_1) and winner_id == id_1:
        return name_1
    if _has(id_2) and winner_id == id_2:
        return name_2
    # The known fighter lost, so the side without an id won.
    return name_2 if _has(id_1) else name_1


def append_log(path: Path, new_rows: pd.DataFrame) -> int:
    """Append rows whose fight_id isn't logged yet. Existing rows are never changed."""
    existing = None
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        new_rows = new_rows[~new_rows["fight_id"].isin(set(existing["fight_id"]))]
    new_rows = new_rows.drop_duplicates("fight_id")
    if existing is not None and new_rows.empty:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if existing is not None:
        existing.to_csv(tmp, index=False, encoding="utf-8")
        new_rows[existing.columns].to_csv(tmp, mode="a", header=False, index=False, encoding="utf-8")
    else:
        new_rows.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return len(new_rows)


def running_metrics(log: pd.DataFrame, recent_events: int = RECENT_EVENTS) -> dict:
    """Accuracy and Brier score (of p_predicted_winner vs whether the pick won), overall and
    over the most recent `recent_events` event dates. Draws/NCs are excluded."""
    def summarise(df):
        if df.empty:
            return {"fights": 0, "accuracy": None, "brier": None}
        y = df["correct"].astype(float).to_numpy()
        p = df["p_predicted_winner"].astype(float).to_numpy()
        return {"fights": int(len(df)), "accuracy": float(y.mean()), "brier": float(np.mean((p - y) ** 2))}

    scored = log[log["correct"].notna() & (log["correct"].astype(str) != "")].copy()
    scored["correct"] = scored["correct"].astype(str).str.lower().map({"true": 1, "false": 0})
    dates = sorted(scored["event_date"].unique())[-recent_events:]
    return {"overall": summarise(scored), "events_overall": int(scored["event_date"].nunique()),
            f"last_{recent_events}_events": summarise(scored[scored["event_date"].isin(dates)]),
            "excluded_draw_nc": int(len(log) - len(scored))}


def run_track(tracked_at: str | None = None) -> dict:
    cfg = load_config()
    tracked_at = tracked_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    preds = load_predictions(resolve_path(cfg["paths"]["predictions_dir"]))
    log_path = resolve_path(cfg["paths"]["tracking_dir"]) / "results_log.csv"

    added = 0
    if len(preds):
        fights = load_processed()["fights"].astype({"fight_id": str, "fighter_1_id": str, "fighter_2_id": str,
                                                    "winner_id": "string", "result": str})
        matched = match_predictions(preds, fights)
        counts = matched["status"].value_counts().to_dict()
        logger.info("Predictions: %d bouts in %d files; status %s", len(preds),
                    preds["prediction_file"].nunique(), counts)
        added = append_log(log_path, build_log_rows(matched, fights, tracked_at))
    else:
        logger.info("No prediction files yet")
    logger.info("Results log: %d new row(s)", added)

    if not log_path.exists():
        logger.info("No tracked results yet")
        return {"added": added, "metrics": None}
    log = pd.read_csv(log_path, dtype=str, keep_default_na=False, na_values=[""])
    metrics = running_metrics(log)
    o, r = metrics["overall"], metrics[f"last_{RECENT_EVENTS}_events"]
    if o["fights"]:
        logger.info("Overall: %d fights / %d events, accuracy %.1f%%, Brier %.4f", o["fights"],
                    metrics["events_overall"], 100 * o["accuracy"], o["brier"])
        logger.info("Last %d events: %d fights, accuracy %.1f%%, Brier %.4f", RECENT_EVENTS, r["fights"],
                    100 * r["accuracy"], r["brier"])
    return {"added": added, "metrics": metrics, "log_rows": len(log)}


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_track()
    except Exception:
        logger.exception("Tracking failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
