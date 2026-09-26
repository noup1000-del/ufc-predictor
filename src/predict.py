"""Stage 5: win probabilities for the next card.

Card source: `data/raw/upcoming_card.json` (manual, primary) else ESPN (see src/upcoming.py).
Features come from `features.matchup_features` -- the same code path as training --
with history from all processed fights before the event date. Both orientations are
scored with models/latest.pkl and combined with `features.symmetric_probability`.

`run_predict_all` fetches the ufc.com schedule (src/upcoming.py), predicts every scheduled
card in data/raw/cards/ and writes the overview page outputs/predictions/index.html.

Usage:
    python -m src.predict                     # manual card file, else ufc.com, else ESPN
    python -m src.predict --card path.json    # a specific card file
    python -m src.predict --all               # every scheduled card + index.html
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import load_processed
from src.config import load_config, resolve_path
from src.features import build_history, matchup_features, swap_orientation, symmetric_probability
from src.http import FetchError
from src.matching import FighterMatcher
from src.report import render_html, render_index
from src.train import load_model
from src.upcoming import (Card, UpcomingCardError, get_upcoming_card, load_aliases, load_scheduled_cards, match_card,
                          slugify, update_schedule)

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["event_date", "weight_class", "fighter_1", "fighter_2", "p_fighter_1", "p_fighter_2",
                  "predicted_winner", "confidence", "model_version", "predicted_at"]
# Appended after the spec columns: IDs let track.py join results on fighter_id, not names.
EXTRA_COLUMNS = ["event_name", "bout_order", "fighter_1_id", "fighter_2_id", "fighter_1_match",
                 "fighter_2_match", "fighter_1_debut", "fighter_2_debut", "key_factors"]

TOP_DRIVERS = 3
MIN_DRIVER_LOGODDS = 0.02  # smaller contributions are noise, not "drivers"

# Driver group -> (label, value format). f1_/f2_/diff_ versions of a feature form one group.
_PCT, _INT, _SIGNED, _DEC, _INCH, _YESNO, _SEC = "pct", "int", "signed", "dec", "inch", "yesno", "sec"
DRIVER_LABELS = {
    "n_fights": ("UFC fights", _INT), "n_wins": ("UFC wins", _INT), "n_losses": ("UFC losses", _INT),
    "win_rate": ("Win rate", _PCT), "streak": ("Streak", _SIGNED),
    "days_since_last_fight": ("Layoff (days)", _INT), "is_debut": ("UFC debut", _YESNO),
    "finish_rate": ("Finish rate", _PCT), "ko_losses": ("KO/TKO losses", _INT),
    "sub_losses": ("Submission losses", _INT), "n_stat_fights": ("Fights with stats", _INT),
    "sig_landed_p15": ("Sig. strikes landed /15m", _DEC), "sig_absorbed_p15": ("Sig. strikes absorbed /15m", _DEC),
    "sig_acc": ("Striking accuracy", _PCT), "sig_def": ("Striking defense", _PCT),
    "kd_p15": ("Knockdowns /15m", _DEC), "td_landed_p15": ("Takedowns /15m", _DEC),
    "td_acc": ("Takedown accuracy", _PCT), "td_def": ("Takedown defense", _PCT),
    "sub_att_p15": ("Sub attempts /15m", _DEC), "ctrl_p15": ("Control time /15m", _SEC),
    "height_in": ("Height", _INCH), "reach_in": ("Reach", _INCH), "age": ("Age", _DEC),
    "first_in_weight_class": ("New to division", _YESNO),
    "weight_class": ("Weight class", None), "stance": ("Stance matchup", None),
    "is_title_fight": ("Title fight", None), "scheduled_rounds": ("Scheduled rounds", None),
}


# --------------------------------------------------------------------------- explainability

def driver_group(column: str) -> str:
    """'f1_sig_def_l3' -> 'sig_def_l3'; stance columns -> 'stance'."""
    if column in ("stance_matchup", "f1_stance", "f2_stance"):
        return "stance"
    return re.sub(r"^(f1_|f2_|diff_)", "", column)


def explain(model, ab: pd.DataFrame, ba: pd.DataFrame) -> pd.DataFrame:
    """Per-bout log-odds contribution of each driver group, positive = favours fighter_1.

    Mirrors the symmetric prediction: 0.5 * (contrib(A,B) - contrib(B,A)). The bias cancels,
    so a row's sum is 0.5 * (logit p(A,B) - logit p(B,A)), whose sign is the sign of
    p_fighter_1 - 0.5."""
    def grouped(c: pd.DataFrame) -> pd.DataFrame:
        c = c.drop(columns="_bias")
        return c.T.groupby(c.columns.map(driver_group)).sum().T

    return (grouped(model.contributions(ab)) - grouped(model.contributions(ba))) / 2


def _fmt(value, kind) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "n/a"
    v = float(value)
    return {_PCT: f"{v:.0%}", _INT: f"{v:.0f}", _SIGNED: f"{v:+.0f}", _DEC: f"{v:.1f}",
            _INCH: f"{v:.0f}\"", _YESNO: "yes" if v else "no", _SEC: f"{v:.0f}s"}[kind]


def driver_values(group: str, row: pd.Series) -> str:
    """Human-readable values behind a driver, fighter_1 vs fighter_2 (raw, pre-imputation)."""
    if group == "stance":
        def stance(v):  # NaN is truthy, so `v or "n/a"` would print "nan"
            return "n/a" if v is None or pd.isna(v) or v == "" else str(v)
        return f"{stance(row.get('f1_stance'))} vs {stance(row.get('f2_stance'))}"
    if group in ("weight_class", "is_title_fight", "scheduled_rounds"):
        v = row.get(group)
        return "n/a" if pd.isna(v) else (str(v) if group == "weight_class" else f"{float(v):.0f}")
    base = group.removesuffix("_l3")
    kind = DRIVER_LABELS.get(base, (group, _DEC))[1]
    return f"{_fmt(row.get(f'f1_{group}'), kind)} vs {_fmt(row.get(f'f2_{group}'), kind)}"


def driver_label(group: str) -> str:
    base = group.removesuffix("_l3")
    label = DRIVER_LABELS.get(base, (base.replace("_", " "), None))[0]
    return f"{label} (last 3)" if group.endswith("_l3") else label


def top_drivers(contrib: pd.Series, row: pd.Series, n: int = TOP_DRIVERS) -> dict:
    """{'fighter_1': [...], 'fighter_2': [...]}, each a list of {group,label,values,logodds}."""
    out = {}
    for side, s in (("fighter_1", contrib[contrib > MIN_DRIVER_LOGODDS].sort_values(ascending=False)),
                    ("fighter_2", contrib[contrib < -MIN_DRIVER_LOGODDS].sort_values())):
        out[side] = [{"group": g, "label": driver_label(g), "values": driver_values(g, row),
                      "logodds": float(v)} for g, v in s.head(n).items()]
    return out


def key_factors_text(drivers: dict, fighter_1: str, fighter_2: str) -> str:
    def part(name, items):
        if not items:
            return f"{name}: none"
        return f"{name}: " + "; ".join(f"{d['label']} {d['values']} ({abs(d['logodds']):.2f})" for d in items)
    return f"{part(fighter_1, drivers['fighter_1'])} | {part(fighter_2, drivers['fighter_2'])}"


def card_pairs(card: Card) -> pd.DataFrame:
    """One row per bout in the card's order (orientation A vs B)."""
    rows = []
    for i, b in enumerate(card.bouts):
        order = b.bout_order or i + 1
        rows.append({
            "bout_order": order,
            "event_date": pd.Timestamp(card.event_date),
            "fighter_1_id": b.fighter_1_id, "fighter_2_id": b.fighter_2_id,
            "fighter_1": b.fighter_1, "fighter_2": b.fighter_2,
            "fighter_1_match": b.fighter_1_match, "fighter_2_match": b.fighter_2_match,
            "weight_class": b.weight_class,
            "is_title_fight": float(b.is_title_fight or 0),
            "scheduled_rounds": float(b.scheduled_rounds or (5 if order == 1 else 3)),
            "card_segment": b.card_segment,
        })
    return pd.DataFrame(rows)


def predict_card(card: Card, tables: dict[str, pd.DataFrame], artifact: dict,
                 predicted_at: str | None = None) -> pd.DataFrame:
    pairs = card_pairs(card)
    history = build_history(tables)
    last_data = tables["fights"]["event_date"].max()
    if pd.Timestamp(card.event_date) <= last_data:
        logger.warning("Card date %s is not after the latest fight in the data (%s); features still only "
                       "use fights strictly before the card date", card.event_date, last_data.date())

    ab = matchup_features(pairs, history)
    ba = matchup_features(swap_orientation(pairs), history)
    model = artifact["model"]
    p1 = symmetric_probability(model.predict_proba(ab), model.predict_proba(ba))
    contrib = explain(model, ab, ba)
    drivers = [top_drivers(contrib.iloc[i], ab.iloc[i]) for i in range(len(pairs))]

    out = pd.DataFrame({
        "event_date": card.event_date,
        "weight_class": pairs["weight_class"],
        "fighter_1": pairs["fighter_1"], "fighter_2": pairs["fighter_2"],
        "p_fighter_1": np.round(p1, 4), "p_fighter_2": np.round(1 - np.round(p1, 4), 4),
        "predicted_winner": np.where(p1 >= 0.5, pairs["fighter_1"], pairs["fighter_2"]),
        "confidence": np.round(np.maximum(p1, 1 - p1), 4),
        "model_version": artifact["metadata"]["model_version"],
        "predicted_at": predicted_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "event_name": card.event_name,
        "bout_order": pairs["bout_order"],
        "fighter_1_id": pairs["fighter_1_id"], "fighter_2_id": pairs["fighter_2_id"],
        "fighter_1_match": pairs["fighter_1_match"], "fighter_2_match": pairs["fighter_2_match"],
        "fighter_1_debut": ab["f1_is_debut"].astype(int), "fighter_2_debut": ab["f2_is_debut"].astype(int),
        "key_factors": [key_factors_text(d, f1, f2) for d, f1, f2 in zip(drivers, pairs["fighter_1"], pairs["fighter_2"])],
        "scheduled_rounds": pairs["scheduled_rounds"].astype(int),
        "card_segment": pairs["card_segment"],   # main/prelims/early_prelims, for the HTML grouping
        "_drivers": drivers,   # structured drivers for the HTML report (not written to CSV)
    })
    return out[OUTPUT_COLUMNS + EXTRA_COLUMNS + ["scheduled_rounds", "card_segment", "_drivers"]]


def format_table(pred: pd.DataFrame) -> str:
    lines = []
    w1 = max(12, pred.fighter_1.str.len().max() + 2)  # room for the debut marker
    w2 = max(12, pred.fighter_2.str.len().max() + 2)
    ww = max(12, pred.weight_class.fillna("").str.len().max())
    head = f"{'#':>2}  {'Weight class':<{ww}}  {'Fighter 1':<{w1}}  {'P1':>5}  {'P2':>5}  {'Fighter 2':<{w2}}  Pick"
    lines += [head, "-" * len(head) + "-" * 20]
    for r in pred.itertuples(index=False):
        tag1 = " *" if r.fighter_1_debut else ""
        tag2 = " *" if r.fighter_2_debut else ""
        p1 = round(r.p_fighter_1 * 100, 1)  # displayed pair always sums to 100.0
        lines.append(f"{r.bout_order:>2}  {str(r.weight_class or ''):<{ww}}  {r.fighter_1 + tag1:<{w1}}  "
                     f"{p1:>4.1f}%  {100 - p1:>4.1f}%  {r.fighter_2 + tag2:<{w2}}  "
                     f"{r.predicted_winner} ({r.confidence:.0%})")
    lines.append("* = UFC debut (no UFC history; stats-based features unknown)")
    return "\n".join(lines)


def run_predict(card_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    cfg = load_config()
    if card_path is not None and not card_path.exists():
        raise UpcomingCardError(f"card file not found: {card_path}")
    tables = load_processed()
    card = get_upcoming_card(fighters=tables["fighters"].astype({"fighter_id": str, "name": str}),
                             card_path=card_path)
    artifact = load_model()
    meta = artifact["metadata"]
    logger.info("Model %s (%s), trained on data up to %s", meta["model_version"], meta["model_type"],
                meta["data_cutoff"])

    return _predict_and_write(card, tables, artifact, resolve_path(cfg["paths"]["predictions_dir"]))


def _warn_matches(card: Card) -> None:
    unmatched = card.unmatched()
    if unmatched:
        logger.warning("%s: no fighter_id for %s: treated as UFC debut(s). If any of them has UFC fights, "
                       "add fighter_1_id/fighter_2_id to the card file.", card.event_name, unmatched)
    for b in card.bouts:
        for side in ("1", "2"):
            if getattr(b, f"fighter_{side}_match") == "fuzzy":
                logger.warning("Fuzzy name match used for %r; check it", getattr(b, f"fighter_{side}"))


def _predict_and_write(card: Card, tables: dict[str, pd.DataFrame], artifact: dict,
                       out_dir: Path, index_link: bool = False) -> tuple[pd.DataFrame, Path]:
    """Predict one matched card; write <date>_<slug>.csv/.html and print the console table."""
    meta = artifact["metadata"]
    _warn_matches(card)
    pred = predict_card(card, tables, artifact)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card.event_date}_{slugify(card.event_name)}.csv"
    _write_atomic(path, pred[OUTPUT_COLUMNS + EXTRA_COLUMNS].to_csv(index=False))
    html_path = path.with_suffix(".html")
    _write_atomic(html_path, render_html(pred, card, meta, index_link=index_link))

    print(f"\n{card.event_name} - {card.event_date}  (source: {card.source}, model: {meta['model_version']})\n")
    print(format_table(pred))
    print(f"\nSaved {path}\nSaved {html_path}\n")
    return pred, path


def run_predict_all(fetch: bool = True) -> list[dict]:
    """Refresh the ufc.com schedule, predict every scheduled card, write index.html.

    If ufc.com refuses (403 / challenge), the card files already in data/raw/cards/ are used
    and the failure is logged; with no card files at all this raises UpcomingCardError.
    Returns one summary dict per scheduled event (the rows of index.html)."""
    cfg = load_config()
    if fetch:
        try:
            update_schedule()
        except (FetchError, UpcomingCardError) as e:
            logger.error("Schedule fetch from ufc.com failed (%s); using existing card files in %s",
                         getattr(e, "reason", e), cfg["upcoming"]["cards_dir"])
    scheduled = load_scheduled_cards()
    if not scheduled:
        raise UpcomingCardError(f"no upcoming card files in {cfg['upcoming']['cards_dir']}; fetch the "
                                "schedule or add a card JSON (see tests/fixtures/upcoming_card.example.json)")

    tables = load_processed()
    fighters = tables["fighters"].astype({"fighter_id": str, "name": str})
    matcher = FighterMatcher.from_fighters(fighters)
    aliases = load_aliases(known_ids=set(fighters.fighter_id))
    artifact = load_model()
    meta = artifact["metadata"]
    logger.info("Model %s (%s), trained on data up to %s; %d scheduled cards", meta["model_version"],
                meta["model_type"], meta["data_cutoff"], len(scheduled))
    out_dir = resolve_path(cfg["paths"]["predictions_dir"])

    entries = []
    for card_file, card in scheduled:
        entry = {"event_name": card.event_name, "event_date": card.event_date, "location": card.location,
                 "event_url": card.event_url, "card_file": card_file.name, "bouts": len(card.bouts),
                 "report": None, "headliner": None, "unmatched": 0, "pred": None,
                 "changes": card.changes, "fetched_at": card.fetched_at}
        if card.bouts:
            card = match_card(card, matcher, aliases)
            pred, path = _predict_and_write(card, tables, artifact, out_dir, index_link=True)
            main = pred.sort_values("bout_order").iloc[0]
            entry.update(report=path.with_suffix(".html").name, unmatched=len(card.unmatched()), pred=pred,
                         headliner={k: main[k] for k in ("fighter_1", "fighter_2", "p_fighter_1", "p_fighter_2",
                                                         "predicted_winner", "confidence", "weight_class")})
        else:
            logger.info("%s (%s): no bouts announced yet; not predicted", card.event_name, card.event_date)
        entries.append(entry)

    index = out_dir / "index.html"
    _write_atomic(index, render_index(entries, meta))
    print(format_schedule(entries))
    print(f"\nSaved {index}\n")
    return entries


def format_schedule(entries: list[dict]) -> str:
    head = f"{'Date':<10}  {'Event':<42}  {'Bouts':>5}  Main event"
    lines = [head, "-" * 110]
    for e in entries:
        h = e["headliner"]
        main = (f"{h['fighter_1']} {h['p_fighter_1']:.0%} vs {h['fighter_2']} {h['p_fighter_2']:.0%}"
                f"  -> {h['predicted_winner']}") if h else "no bouts announced yet"
        lines.append(f"{e['event_date']:<10}  {e['event_name'][:42]:<42}  {e['bouts']:>5}  {main}")
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--card", type=Path, help="card JSON file (default: config upcoming.manual_card_file)")
    p.add_argument("--all", action="store_true", help="fetch the ufc.com schedule and predict every card")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # non-UTF-8 console: don't crash on "ć"
    try:
        if args.all:
            run_predict_all()
        else:
            run_predict(args.card)
    except UpcomingCardError as e:
        logger.error("%s", e)
        return 1
    except Exception:
        logger.exception("Prediction failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
