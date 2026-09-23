"""Stage 2: raw tables -> typed, validated tables in data/processed/.

- Drops fights before `clean.min_date` (config) and their stats/events.
- Keeps draws/NCs in the fight history but marks them `is_target = False`
  (only fights with a winner are training targets).
- Adds `method_group` (ko_tko / submission / decision / dq / other) and `has_stats`.
- Validates and writes a report to data/processed/clean_report.json.

Downstream code must read processed tables with `load_processed()` so dtypes are
defined in exactly one place.

Usage: python -m src.clean
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

# Column -> logical type. The order here is the column order of the processed CSVs.
SCHEMA: dict[str, dict[str, str]] = {
    "events": {
        "event_id": "string", "event_name": "string", "event_date": "date", "location": "string",
        "scraped_at": "timestamp",
    },
    "fights": {
        "fight_id": "string", "event_id": "string", "event_date": "date", "bout_order": "int",
        "fighter_1_id": "string", "fighter_2_id": "string", "fighter_1_name": "string",
        "fighter_2_name": "string", "winner_id": "string", "result": "category", "method": "string",
        "method_group": "category", "end_round": "int", "end_time_sec": "int",
        "scheduled_rounds": "int", "weight_class": "category", "is_title_fight": "bool",
        "is_target": "bool", "has_stats": "bool",
    },
    "fight_stats": {
        "fight_id": "string", "fighter_id": "string", "opponent_id": "string", "kd": "int",
        "sig_str_landed": "int", "sig_str_attempted": "int", "total_str_landed": "int",
        "total_str_attempted": "int", "td_landed": "int", "td_attempted": "int", "sub_att": "int",
        "reversals": "int", "ctrl_sec": "int", "sig_head_landed": "int", "sig_body_landed": "int",
        "sig_leg_landed": "int", "sig_distance_landed": "int", "sig_clinch_landed": "int",
        "sig_ground_landed": "int", "fight_duration_sec": "int",
    },
    "fighters": {
        "fighter_id": "string", "name": "string", "nickname": "string", "height_in": "float",
        "weight_lb": "float", "reach_in": "float", "stance": "category", "dob": "date",
        "scraped_at": "timestamp",
    },
}
# Fights before min_date: history only (records/experience), never targets, no stats.
SCHEMA["fights_prior"] = dict(SCHEMA["fights"])
KEYS = {"events": ["event_id"], "fights": ["fight_id"], "fight_stats": ["fight_id", "fighter_id"],
        "fighters": ["fighter_id"], "fights_prior": ["fight_id"]}
RAW_TABLES = ["events", "fights", "fight_stats", "fighters"]
TABLES = list(SCHEMA)

MIN_AGE, MAX_AGE = 18, 50


def method_group(method) -> str | None:
    if method is None or pd.isna(method):
        return None
    m = str(method).lower()
    if m.startswith("decision"):
        return "decision"
    if m.startswith(("ko", "tko")):  # "KO/TKO", "TKO - Doctor's Stoppage"
        return "ko_tko"
    if m.startswith("submission"):
        return "submission"
    if m == "dq":
        return "dq"
    return "other"  # Overturned, Could Not Continue, Other


# --------------------------------------------------------------------------- typing

class ParseFailures(dict):
    """(table, column) -> number of non-empty values that failed to parse."""


def apply_types(table: str, df: pd.DataFrame, failures: ParseFailures | None = None) -> pd.DataFrame:
    """Cast string columns to the logical types in SCHEMA. Unparseable values become NA
    and are counted in `failures`."""
    out = pd.DataFrame(index=df.index)
    for col, kind in SCHEMA[table].items():
        s = df[col]
        present = s.notna() & (s.astype("string").str.strip() != "")
        if kind == "string":
            t = s.astype("string")
        elif kind == "category":
            t = s.astype("string").astype("category")
        elif kind == "int":
            f = pd.to_numeric(s, errors="coerce")
            bad_frac = f.notna() & (f % 1 != 0)
            f[bad_frac] = pd.NA
            t = f.astype("Int64")
        elif kind == "float":
            t = pd.to_numeric(s, errors="coerce").astype("float64")
        elif kind == "bool":
            t = s.astype("string").str.strip().str.lower().map(
                {"1": True, "true": True, "0": False, "false": False}).astype("boolean")
        elif kind == "date":
            t = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
        elif kind == "timestamp":
            t = pd.to_datetime(s, utc=True, errors="coerce", format="ISO8601")
        else:
            raise ValueError(f"unknown type {kind!r} for {table}.{col}")
        if failures is not None:
            n = int((present & t.isna()).sum())
            if n:
                failures[(table, col)] = n
        out[col] = t
    return out


def _to_csv_frame(table: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(SCHEMA[table])].copy()
    for col, kind in SCHEMA[table].items():
        if kind == "date":
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif kind == "timestamp":
            out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        elif kind == "bool":
            out[col] = out[col].astype("Int64")
    return out


def load_processed(processed_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the processed tables with their declared dtypes."""
    processed_dir = processed_dir or resolve_path(load_config()["paths"]["processed_dir"])
    tables = {}
    for t in TABLES:
        raw = pd.read_csv(processed_dir / f"{t}.csv", dtype=str, keep_default_na=False, na_values=[""])
        tables[t] = apply_types(t, raw)
    return tables


def _write_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- cleaning

def clean_tables(raw: dict[str, pd.DataFrame], min_date: str) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return (typed processed tables, report). `raw` holds the raw CSVs read as strings."""
    failures = ParseFailures()
    dropped: list[dict] = []

    def drop(table, reason, n):
        if n:
            dropped.append({"table": table, "reason": reason, "rows": int(n)})

    raw_fights = raw["fights"].copy()
    raw_fights["method_group"] = raw_fights["method"].map(method_group)
    raw_fights["is_target"] = (raw_fights["result"] == "win").map({True: "1", False: "0"})
    raw_fights["has_stats"] = "1"  # set properly below
    typed = {t: apply_types(t, raw_fights if t == "fights" else raw[t], failures) for t in RAW_TABLES}

    # Rows that can't be placed in time are unusable.
    for t in ("fights", "events"):
        bad = typed[t]["event_date"].isna()
        drop(t, "event_date missing or unparseable", bad.sum())
        typed[t] = typed[t][~bad]

    # Duplicates (should not happen after ingest; keep the first occurrence).
    for t in RAW_TABLES:
        dup = typed[t].duplicated(KEYS[t])
        drop(t, "duplicate key", dup.sum())
        typed[t] = typed[t][~dup]
    events, fights, stats, fighters = (typed[t] for t in ("events", "fights", "fight_stats", "fighters"))

    cutoff = pd.Timestamp(min_date)
    before = fights["event_date"] < cutoff
    drop("fights", f"before min_date {min_date} (moved to fights_prior)", before.sum())
    prior = fights[before].copy()
    prior["is_target"] = pd.array([False] * len(prior), dtype="boolean")
    prior["has_stats"] = pd.array([False] * len(prior), dtype="boolean")
    fights = fights[~before]

    orphan = ~stats["fight_id"].isin(set(fights["fight_id"]))
    drop("fight_stats", "fight dropped (before min_date or invalid)", orphan.sum())
    stats = stats[~orphan]

    # A fight's stats are only usable as a pair: one row per fighter, matching the fight.
    pair = stats.merge(fights[["fight_id", "fighter_1_id", "fighter_2_id"]], on="fight_id", how="left")
    wrong_fighter = (pair["fighter_id"] != pair["fighter_1_id"]) & (pair["fighter_id"] != pair["fighter_2_id"])
    wrong_opponent = pair["opponent_id"] != pair["fighter_1_id"].where(
        pair["fighter_id"] == pair["fighter_2_id"], pair["fighter_2_id"])
    bad_rows = (wrong_fighter | wrong_opponent).to_numpy()
    drop("fight_stats", "fighter/opponent not matching the fight", bad_rows.sum())
    stats = stats[~bad_rows]
    counts = stats.groupby("fight_id").size()
    n_rows = fights["fight_id"].map(counts).fillna(0).astype(int)
    not_pair = set(fights.loc[(n_rows != 2) & (n_rows != 0), "fight_id"])
    drop("fight_stats", "fight has 1 or >2 stat rows (fight kept, stats removed)",
         stats["fight_id"].isin(not_pair).sum())
    stats = stats[~stats["fight_id"].isin(not_pair)]
    fights = fights.copy()
    fights["has_stats"] = fights["fight_id"].isin(set(stats["fight_id"])).astype("boolean")

    events = events.copy()
    no_fights = ~events["event_id"].isin(set(fights["event_id"]))
    drop("events", "no fights on or after min_date", no_fights.sum())
    events = events[~no_fights]

    tables = {
        "events": events.sort_values("event_date").reset_index(drop=True),
        "fights": fights.sort_values(["event_date", "event_id", "bout_order"]).reset_index(drop=True),
        "fight_stats": stats.reset_index(drop=True),
        "fighters": fighters.sort_values("fighter_id").reset_index(drop=True),
        "fights_prior": prior.sort_values(["event_date", "event_id", "bout_order"]).reset_index(drop=True),
    }
    report = {
        "min_date": min_date,
        "row_counts": {t: {"raw": len(raw[t]) if t in raw else None, "processed": len(tables[t])}
                       for t in TABLES},
        "dropped": dropped,
        "parse_failures": [{"table": t, "column": c, "rows": n} for (t, c), n in failures.items()],
        "checks": validate_processed(tables),
        "targets": {
            "is_target": int(tables["fights"]["is_target"].sum()),
            "excluded_draw": int((tables["fights"]["result"] == "draw").sum()),
            "excluded_nc": int((tables["fights"]["result"] == "nc").sum()),
        },
        "dtypes": {t: {c: str(d) for c, d in tables[t].dtypes.items()} for t in TABLES},
    }
    return tables, report


# --------------------------------------------------------------------------- validation

def validate_processed(t: dict[str, pd.DataFrame]) -> list[dict]:
    """Checks on typed tables. severity 'error' fails the stage; 'warning' is reported only."""
    ev, fi, st, fr = t["events"], t["fights"], t["fight_stats"], t["fighters"]
    checks = []

    def check(name, bad: pd.DataFrame, cols, severity="error"):
        ex = bad[cols].head(3).astype(str).to_dict("records") if len(bad) else []
        checks.append({"check": name, "severity": severity, "violations": int(len(bad)), "examples": ex})

    for name, df in t.items():
        check(f"{name}: duplicate keys", df[df.duplicated(KEYS[name], keep=False)], KEYS[name])
    check("events: unparseable event_date", ev[ev.event_date.isna()], ["event_id"])
    check("events: unparseable scraped_at", ev[ev.scraped_at.isna()], ["event_id"])
    check("fighters: unparseable scraped_at", fr[fr.scraped_at.isna()], ["fighter_id"])
    check("fights: unparseable event_date", fi[fi.event_date.isna()], ["fight_id"])

    counts = fi["fight_id"].map(st.groupby("fight_id").size()).fillna(0).astype(int)
    check("fights with stats: not exactly 2 stat rows", fi[fi.has_stats & (counts != 2)], ["fight_id"])
    check("fights without stats", fi[~fi.has_stats], ["fight_id", "event_date"], "warning")
    check("fight_stats: fight not in fights", st[~st.fight_id.isin(set(fi.fight_id))], ["fight_id"])

    check("fights: event not in events", fi[~fi.event_id.isin(set(ev.event_id))], ["fight_id", "event_id"])
    ev_date = fi.event_id.map(dict(zip(ev.event_id, ev.event_date)))
    check("fights: event_date differs from events", fi[fi.event_date != ev_date], ["fight_id"])
    known = set(fr.fighter_id)
    check("fights: fighter not in fighters",
          fi[~fi.fighter_1_id.isin(known) | ~fi.fighter_2_id.isin(known)], ["fight_id"])
    check("fights: same fighter on both sides", fi[fi.fighter_1_id == fi.fighter_2_id], ["fight_id"])

    win = fi.result == "win"
    check("fights: win without valid winner_id",
          fi[win & (fi.winner_id != fi.fighter_1_id) & (fi.winner_id != fi.fighter_2_id)], ["fight_id"])
    check("fights: draw/nc with winner_id", fi[~win & fi.winner_id.notna()], ["fight_id", "result"])
    check("fights: is_target != (result == win)", fi[fi.is_target != win], ["fight_id", "result"])
    check("fights: method_group missing", fi[fi.method_group.isna()], ["fight_id", "method"], "warning")

    long = pd.concat([fi[["fight_id", "event_date", "fighter_1_id"]].rename(columns={"fighter_1_id": "fid"}),
                      fi[["fight_id", "event_date", "fighter_2_id"]].rename(columns={"fighter_2_id": "fid"})])
    check("fighter in more than one fight on the same date",
          long[long.duplicated(["fid", "event_date"], keep=False)], ["fid", "event_date", "fight_id"])

    pr = t["fights_prior"]
    check("fights_prior: overlaps fights", pr[pr.fight_id.isin(set(fi.fight_id))], ["fight_id"])
    if len(fi):
        check("fights_prior: not strictly before fights", pr[pr.event_date >= fi.event_date.min()], ["fight_id"])
    check("fights_prior: fighter not in fighters",
          pr[~pr.fighter_1_id.isin(known) | ~pr.fighter_2_id.isin(known)], ["fight_id"])
    pwin = pr.result == "win"
    check("fights_prior: win without valid winner_id",
          pr[pwin & (pr.winner_id != pr.fighter_1_id) & (pr.winner_id != pr.fighter_2_id)], ["fight_id"])
    check("fights_prior: marked as target", pr[pr.is_target.fillna(False)], ["fight_id"])

    dob = long.fid.map(dict(zip(fr.fighter_id, fr.dob)))
    age = (long.event_date - dob).dt.days / 365.25
    check(f"age at fight outside {MIN_AGE}-{MAX_AGE}",
          long[age.notna() & ((age < MIN_AGE) | (age > MAX_AGE))], ["fid", "event_date"], "warning")
    return checks


# --------------------------------------------------------------------------- report / entry point

def log_report(report: dict) -> int:
    """Log the report; return the number of failed error-level checks."""
    logger.info("min_date = %s", report["min_date"])
    logger.info("%-12s %8s %10s", "table", "raw", "processed")
    for t, c in report["row_counts"].items():
        logger.info("%-12s %8s %10d", t, "-" if c["raw"] is None else c["raw"], c["processed"])
    for d in report["dropped"]:
        logger.info("dropped %-12s %6d  %s", d["table"], d["rows"], d["reason"])
    for p in report["parse_failures"]:
        logger.warning("parse failures %s.%s: %d", p["table"], p["column"], p["rows"])
    logger.info("targets: %s", report["targets"])
    errors = 0
    for c in report["checks"]:
        ok = c["violations"] == 0
        errors += (not ok) and c["severity"] == "error"
        level = logging.INFO if ok else (logging.ERROR if c["severity"] == "error" else logging.WARNING)
        logger.log(level, "[%s] %s: %d%s", "ok" if ok else c["severity"], c["check"], c["violations"],
                   f" e.g. {c['examples']}" if c["examples"] else "")
    return errors


def run_clean() -> dict:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    raw = {t: pd.read_csv(raw_dir / f"{t}.csv", dtype=str, keep_default_na=False, na_values=[""])
           for t in RAW_TABLES}
    tables, report = clean_tables(raw, str(cfg["clean"]["min_date"]))
    for t, df in tables.items():
        _write_atomic_csv(_to_csv_frame(t, df), out_dir / f"{t}.csv")
    (out_dir / "clean_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        errors = log_report(run_clean())
    except Exception:
        logger.exception("Clean failed")
        return 1
    if errors:
        logger.error("Clean validation: %d error-level check(s) failed", errors)
        return 1
    logger.info("Clean validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
