"""Import the Greco1899 ufcstats archive into the raw schemas in data/raw/.

This is the only module (besides upcoming-card matching) that joins on fighter
names, because the archive's results/stats files identify fighters by name.

Usage:
    python -m src.ingest                 # fetch latest source commit, merge new rows, validate
    python -m src.ingest --full          # rebuild raw tables from the newest cached source
    python -m src.ingest --validate-only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.http import FetchError, HttpClient, get_client
from src.matching import FighterMatcher, canonical_weight_class, fighter_info, normalize_name

logger = logging.getLogger(__name__)

SCHEMAS = {
    "fighters": ["fighter_id", "name", "nickname", "height_in", "weight_lb", "reach_in",
                 "stance", "dob", "scraped_at"],
    "fights": ["fight_id", "event_id", "event_date", "bout_order", "fighter_1_id", "fighter_2_id",
               "fighter_1_name", "fighter_2_name", "winner_id", "result", "method", "end_round",
               "end_time_sec", "scheduled_rounds", "weight_class", "is_title_fight"],
    "fight_stats": ["fight_id", "fighter_id", "opponent_id", "kd", "sig_str_landed",
                    "sig_str_attempted", "total_str_landed", "total_str_attempted", "td_landed",
                    "td_attempted", "sub_att", "reversals", "ctrl_sec", "sig_head_landed",
                    "sig_body_landed", "sig_leg_landed", "sig_distance_landed",
                    "sig_clinch_landed", "sig_ground_landed", "fight_duration_sec"],
    "events": ["event_id", "event_name", "event_date", "location", "scraped_at"],
}
KEYS = {
    "fighters": ["fighter_id"],
    "fights": ["fight_id"],
    "fight_stats": ["fight_id", "fighter_id"],
    "events": ["event_id"],
}
# Written in this order so an event row only appears once its fights exist.
WRITE_ORDER = ["fighters", "fights", "fight_stats", "events"]
DROPPED_COLUMNS = ["fight_id", "event", "bout", "reason"]
OVERRIDE_COLUMNS = ["fight_id", "name", "fighter_id", "note"]

OUTCOMES = {"W/L": "f1", "L/W": "f2", "D/D": "draw", "NC/NC": "nc"}
_MISSING = {"", "--", "---"}


# --------------------------------------------------------------------------- parsers

def _is_missing(s) -> bool:
    return s is None or (isinstance(s, float) and math.isnan(s)) or str(s).strip() in _MISSING


def parse_of(s) -> tuple[float, float]:
    """'12 of 30' -> (12, 30); missing -> (nan, nan)."""
    if _is_missing(s):
        return (np.nan, np.nan)
    m = re.fullmatch(r"\s*(\d+)\s+of\s+(\d+)\s*", str(s))
    if not m:
        raise ValueError(f"not an 'x of y' value: {s!r}")
    return (float(m.group(1)), float(m.group(2)))


def parse_mss(s) -> float:
    """'4:32' -> 272; missing -> nan."""
    if _is_missing(s):
        return np.nan
    m = re.fullmatch(r"\s*(\d+):(\d{2})\s*", str(s))
    if not m:
        raise ValueError(f"not an m:ss value: {s!r}")
    return float(int(m.group(1)) * 60 + int(m.group(2)))


def parse_height(s) -> float:
    """5' 11" -> 71; missing -> nan."""
    if _is_missing(s):
        return np.nan
    m = re.fullmatch(r"\s*(\d+)'\s*(\d+(?:\.\d+)?)?\"?\s*", str(s))
    if not m:
        raise ValueError(f"not a height: {s!r}")
    return float(int(m.group(1)) * 12 + float(m.group(2) or 0))


def parse_reach(s) -> float:
    """72" -> 72; missing -> nan."""
    if _is_missing(s):
        return np.nan
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\"?\s*", str(s))
    if not m:
        raise ValueError(f"not a reach: {s!r}")
    return float(m.group(1))


def parse_weight(s) -> float:
    """'155 lbs.' -> 155; missing -> nan."""
    if _is_missing(s):
        return np.nan
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*lbs?\.?\s*", str(s))
    if not m:
        raise ValueError(f"not a weight: {s!r}")
    return float(m.group(1))


def parse_int(s) -> float:
    """'3' or '3.0' -> 3.0; missing -> nan."""
    if _is_missing(s):
        return np.nan
    m = re.fullmatch(r"\s*(\d+)(?:\.0+)?\s*", str(s))
    if not m:
        raise ValueError(f"not an integer: {s!r}")
    return float(m.group(1))


def parse_date(s) -> str | None:
    """'September 19, 2026' / 'Jul 13, 1978' / 'Jul. 13, 1978' -> 'YYYY-MM-DD'; missing -> None."""
    if _is_missing(s):
        return None
    text = str(s).strip().replace(".", "")
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"not a date: {s!r}")


def extract_id(url) -> str | None:
    if _is_missing(url):
        return None
    m = re.search(r"/(?:event|fight|fighter)-details/([0-9a-f]+)", str(url))
    return m.group(1) if m else None


def parse_time_format(s) -> tuple[float, list[int] | None]:
    """'3 Rnd (5-5-5)' -> (3, [300, 300, 300]); '1 Rnd + OT (15-3)' -> (2, [900, 180]);
    'No Time Limit' -> (nan, None). Overtime periods count as scheduled rounds."""
    if _is_missing(s):
        return (np.nan, None)
    text = str(s).strip()
    p = re.search(r"\(([\d\-\s]+)\)", text)
    if p:
        lengths = [int(x) * 60 for x in re.findall(r"\d+", p.group(1))]
        return (float(len(lengths)), lengths)
    m = re.match(r"(\d+)\s*Rnd", text)
    return (float(m.group(1)) if m else np.nan, None)


def fight_duration(end_round, end_time_sec, round_lengths) -> float:
    if pd.isna(end_round) or pd.isna(end_time_sec):
        return np.nan
    r = int(end_round)
    if round_lengths is None:
        return float(end_time_sec) if r == 1 else np.nan
    if r > len(round_lengths):
        return np.nan
    return float(sum(round_lengths[: r - 1]) + end_time_sec)


# --------------------------------------------------------------------------- source

def latest_commit_sha(client: HttpClient, repo: str, branch: str) -> str:
    data = json.loads(client.get(f"https://api.github.com/repos/{repo}/commits/{branch}"))
    return data["sha"]


def _complete_source_dirs(base: Path, files: list[str]) -> list[Path]:
    if not base.exists():
        return []
    dirs = [d for d in base.iterdir() if d.is_dir() and all((d / f"{f}.csv").exists() for f in files)]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def fetch_source(client: HttpClient, cfg: dict, offline: bool = False) -> tuple[str, Path]:
    """Return (commit sha, directory) of the source CSVs, downloading if needed."""
    src_cfg = cfg["source"]
    base = resolve_path(src_cfg["source_dir"]) / "greco"
    files = src_cfg["files"]
    cached = _complete_source_dirs(base, files)

    if offline:
        if not cached:
            raise RuntimeError(f"No cached source in {base}; run without --full first")
        return cached[0].name, cached[0]

    try:
        sha = latest_commit_sha(client, src_cfg["repo"], src_cfg["branch"])
    except (FetchError, KeyError, ValueError) as e:
        if not cached:
            raise
        logger.warning("Could not get latest commit (%s); using cached source %s", e, cached[0].name)
        return cached[0].name, cached[0]

    d = base / sha
    for f in files:
        url = f"https://raw.githubusercontent.com/{src_cfg['repo']}/{sha}/{f}.csv"
        client.download(url, d / f"{f}.csv")
    logger.info("Source commit %s in %s", sha[:10], d)
    return sha, d


def load_source(d: Path, files: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for f in files:
        df = pd.read_csv(d / f"{f}.csv", dtype=str, keep_default_na=False, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        out[f] = df.apply(lambda col: col.str.strip())
    return out


# --------------------------------------------------------------------------- builders

def build_fighters(src: dict[str, pd.DataFrame], scraped_at: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """Return the fighters table and all (fighter_id, name) pairs for the name index."""
    det = src["ufc_fighter_details"].copy()
    tott = src["ufc_fighter_tott"].copy()
    det["fighter_id"] = det["URL"].map(extract_id)
    tott["fighter_id"] = tott["URL"].map(extract_id)
    det["det_name"] = (det["FIRST"] + " " + det["LAST"]).str.strip()

    for name, df in (("fighter_details", det), ("fighter_tott", tott)):
        bad = df["fighter_id"].isna().sum()
        if bad:
            logger.warning("%d %s rows without a fighter URL skipped", bad, name)
        dups = df["fighter_id"].duplicated() & df["fighter_id"].notna()
        if dups.any():
            logger.warning("%d duplicate fighter ids in %s, keeping first", dups.sum(), name)
    det = det[det["fighter_id"].notna()].drop_duplicates("fighter_id")
    tott = tott[tott["fighter_id"].notna()].drop_duplicates("fighter_id")

    m = det[["fighter_id", "det_name", "NICKNAME"]].merge(
        tott[["fighter_id", "FIGHTER", "HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB"]],
        on="fighter_id", how="outer")
    m = m.fillna("")

    rows = []
    for r in m.itertuples(index=False):
        try:
            rows.append({
                "fighter_id": r.fighter_id,
                "name": r.det_name or r.FIGHTER,
                "nickname": r.NICKNAME or None,
                "height_in": parse_height(r.HEIGHT),
                "weight_lb": parse_weight(r.WEIGHT),
                "reach_in": parse_reach(r.REACH),
                "stance": r.STANCE or None,
                "dob": parse_date(r.DOB),
                "scraped_at": scraped_at,
            })
        except ValueError as e:
            logger.error("fighter %s: %s", r.fighter_id, e)
    fighters = pd.DataFrame(rows, columns=SCHEMAS["fighters"])

    pairs = list(zip(det["fighter_id"], det["det_name"])) + list(zip(tott["fighter_id"], tott["FIGHTER"]))
    return fighters, pairs


def build_events(src: dict[str, pd.DataFrame], scraped_at: str, today: date) -> pd.DataFrame:
    ev = src["ufc_event_details"]
    rows = []
    for r in ev.itertuples(index=False):
        eid = extract_id(r.URL)
        try:
            d = parse_date(r.DATE)
        except ValueError as e:
            logger.error("event %s: %s", eid, e)
            continue
        if not eid or not d:
            logger.error("event row without id/date skipped: %r", r.EVENT)
            continue
        if d > today.isoformat():
            logger.info("skipping future event %s (%s)", r.EVENT, d)
            continue
        rows.append({"event_id": eid, "event_name": r.EVENT, "event_date": d,
                     "location": r.LOCATION or None, "scraped_at": scraped_at})
    events = pd.DataFrame(rows, columns=SCHEMAS["events"])
    dup = events["event_id"].duplicated()
    if dup.any():
        logger.warning("%d duplicate event ids, keeping first", dup.sum())
        events = events[~dup]
    return events


def load_overrides(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {(r.fight_id.strip(), normalize_name(r.name)): r.fighter_id.strip()
            for r in df.itertuples(index=False) if r.fight_id and r.fighter_id}


def split_bout(bout: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+vs\.?\s+", bout, maxsplit=1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        return None
    return parts[0].strip(), parts[1].strip()


def build_fights(src: dict[str, pd.DataFrame], events: pd.DataFrame, matcher: FighterMatcher,
                 overrides: dict[tuple[str, str], str]):
    """Return (fights, per-fight name->id map, dropped rows, bout-key->fight_id map)."""
    fr = src["ufc_fight_results"].rename(columns={"TIME FORMAT": "TIME_FORMAT"})
    fr["fight_id"] = fr["URL"].map(extract_id)
    ev_ids = dict(zip(events["event_name"], events["event_id"]))
    ev_dates = dict(zip(events["event_id"], events["event_date"]))
    fr["event_id"] = fr["EVENT"].map(ev_ids)

    # (EVENT, BOUT) -> fight_id, over *all* rows (incl. stale event names), for the stats join.
    bout_keys: dict[tuple[str, str], set[str]] = {}
    for r in fr.itertuples(index=False):
        if r.fight_id:
            bout_keys.setdefault((r.EVENT, r.BOUT), set()).add(r.fight_id)

    dropped = []
    no_id = fr["fight_id"].isna()
    for r in fr[no_id].itertuples(index=False):
        dropped.append({"fight_id": "", "event": r.EVENT, "bout": r.BOUT, "reason": "no fight URL"})
    fr = fr[~no_id]

    # Renamed events: the same fight under an old and a new event name.
    has_event = fr["event_id"].notna()
    stale = ~has_event & fr["fight_id"].isin(set(fr.loc[has_event, "fight_id"]))
    if stale.any():
        logger.info("%d fight rows listed under a stale event name dropped (kept the current name)", stale.sum())
    fr = fr[~stale]
    for r in fr[fr["event_id"].isna()].itertuples(index=False):
        dropped.append({"fight_id": r.fight_id, "event": r.EVENT, "bout": r.BOUT,
                        "reason": "event not in events file (or future event)"})
    fr = fr[fr["event_id"].notna()]
    dup = fr["fight_id"].duplicated()
    for r in fr[dup].itertuples(index=False):
        dropped.append({"fight_id": r.fight_id, "event": r.EVENT, "bout": r.BOUT,
                        "reason": "duplicate fight id"})
    fr = fr[~dup].copy()
    fr["bout_order"] = fr.groupby("event_id", sort=False).cumcount() + 1

    rows, name_map = [], {}
    for r in fr.itertuples(index=False):
        try:
            names = split_bout(r.BOUT)
            if names is None:
                raise _Drop(f"cannot split bout {r.BOUT!r}")
            event_date = ev_dates[r.event_id]
            on_date = date.fromisoformat(event_date)
            ids = []
            for name in names:
                fid = overrides.get((r.fight_id, normalize_name(name)))
                if fid is None:
                    m = matcher.match(name, weight_class=r.WEIGHTCLASS, on_date=on_date, fuzzy=False)
                    if not m.ok:
                        cands = f" candidates={','.join(m.candidates)}" if m.candidates else ""
                        raise _Drop(f"fighter name {m.method}: {name!r}{cands}")
                    fid = m.fighter_id
                ids.append(fid)
            if ids[0] == ids[1]:
                raise _Drop(f"both names resolved to {ids[0]}")

            outcome = OUTCOMES.get(r.OUTCOME)
            if outcome is None:
                raise _Drop(f"unknown outcome {r.OUTCOME!r}")
            winner = {"f1": ids[0], "f2": ids[1]}.get(outcome)
            result = "win" if winner else outcome

            end_round = parse_int(r.ROUND)
            end_time = parse_mss(r.TIME)
            scheduled, lengths = parse_time_format(r.TIME_FORMAT)
            wc_raw = r.WEIGHTCLASS
            rows.append({
                "fight_id": r.fight_id,
                "event_id": r.event_id,
                "event_date": event_date,
                "bout_order": r.bout_order,
                "fighter_1_id": ids[0],
                "fighter_2_id": ids[1],
                "fighter_1_name": names[0],
                "fighter_2_name": names[1],
                "winner_id": winner,
                "result": result,
                "method": r.METHOD or None,
                "end_round": end_round,
                "end_time_sec": end_time,
                "scheduled_rounds": scheduled,
                "weight_class": canonical_weight_class(wc_raw),
                "is_title_fight": int("title bout" in wc_raw.lower() and "tournament" not in wc_raw.lower()),
                "_duration": fight_duration(end_round, end_time, lengths),
            })
            for name, fid in zip(names, ids):
                name_map[(r.fight_id, normalize_name(name))] = fid
        except _Drop as e:
            dropped.append({"fight_id": r.fight_id, "event": r.EVENT, "bout": r.BOUT, "reason": str(e)})
        except (ValueError, KeyError) as e:
            logger.error("fight %s: %s", r.fight_id, e)
            dropped.append({"fight_id": r.fight_id, "event": r.EVENT, "bout": r.BOUT,
                            "reason": f"parse error: {e}"})

    fights = pd.DataFrame(rows, columns=SCHEMAS["fights"] + ["_duration"])
    return fights, name_map, pd.DataFrame(dropped, columns=DROPPED_COLUMNS), bout_keys


class _Drop(Exception):
    """A fight that can't be mapped; recorded in ingest_dropped.csv."""


_STAT_OF = {"SIG.STR.": "sig_str", "TOTAL STR.": "total_str", "TD": "td"}
_STAT_LANDED = {"HEAD": "sig_head_landed", "BODY": "sig_body_landed", "LEG": "sig_leg_landed",
                "DISTANCE": "sig_distance_landed", "CLINCH": "sig_clinch_landed",
                "GROUND": "sig_ground_landed"}
_STAT_INT = {"KD": "kd", "SUB.ATT": "sub_att", "REV.": "reversals"}
_STAT_SUM_COLS = (["kd", "sub_att", "reversals", "ctrl_sec"]
                  + [f"{v}_{s}" for v in _STAT_OF.values() for s in ("landed", "attempted")]
                  + list(_STAT_LANDED.values()))


def build_fight_stats(src: dict[str, pd.DataFrame], fights: pd.DataFrame,
                      name_map: dict[tuple[str, str], str],
                      bout_keys: dict[tuple[str, str], set[str]]) -> pd.DataFrame:
    """Sum per-round rows into one fight-total row per fighter."""
    fs = src["ufc_fight_stats"]
    fs = fs[fs["ROUND"] != ""].copy()
    kept = set(fights["fight_id"])

    def fight_for(key):
        ids = bout_keys.get(key)
        if not ids:
            return "missing"
        return next(iter(ids)) if len(ids) == 1 else "ambiguous"

    fs["fight_id"] = [fight_for(k) for k in zip(fs["EVENT"], fs["BOUT"])]
    for status in ("missing", "ambiguous"):
        n = (fs["fight_id"] == status).sum()
        if n:
            logger.warning("%d stat rows with %s (EVENT, BOUT) -> fight mapping skipped", n, status)
    fs = fs[fs["fight_id"].isin(kept)]

    fs["norm"] = fs["FIGHTER"].map(normalize_name)
    before = len(fs)
    fs = fs.drop_duplicates(["fight_id", "norm", "ROUND"])
    if before - len(fs):
        logger.info("%d duplicate per-round stat rows removed", before - len(fs))

    fs["fighter_id"] = [name_map.get(k) for k in zip(fs["fight_id"], fs["norm"])]
    unmatched = fs["fighter_id"].isna()
    if unmatched.any():
        logger.warning("%d stat rows whose FIGHTER is not one of the bout's two names skipped "
                       "(fights: %s)", unmatched.sum(), sorted(set(fs.loc[unmatched, "fight_id"]))[:10])
    fs = fs[~unmatched]

    parsed = pd.DataFrame({"fight_id": fs["fight_id"].values, "fighter_id": fs["fighter_id"].values})
    for col, out in _STAT_INT.items():
        parsed[out] = fs[col].map(parse_int).values
    for col, out in _STAT_OF.items():
        pairs = fs[col].map(parse_of)
        parsed[f"{out}_landed"] = [p[0] for p in pairs]
        parsed[f"{out}_attempted"] = [p[1] for p in pairs]
    for col, out in _STAT_LANDED.items():
        parsed[out] = [p[0] for p in fs[col].map(parse_of)]
    parsed["ctrl_sec"] = fs["CTRL"].map(parse_mss).values

    totals = parsed.groupby(["fight_id", "fighter_id"], sort=False)[_STAT_SUM_COLS].sum(min_count=1).reset_index()
    f = fights.set_index("fight_id")
    totals["opponent_id"] = [
        f.at[fid, "fighter_2_id"] if f.at[fid, "fighter_1_id"] == pid else f.at[fid, "fighter_1_id"]
        for fid, pid in zip(totals["fight_id"], totals["fighter_id"])
    ]
    totals["fight_duration_sec"] = totals["fight_id"].map(f["_duration"])
    return totals[SCHEMAS["fight_stats"]]


# --------------------------------------------------------------------------- storage

_INT_COLUMNS = {
    "fights": ["bout_order", "end_round", "end_time_sec", "scheduled_rounds", "is_title_fight"],
    "fight_stats": [c for c in SCHEMAS["fight_stats"] if c not in ("fight_id", "fighter_id", "opponent_id")],
}


def _format_for_csv(table: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df[SCHEMAS[table]].copy()
    for c in _INT_COLUMNS.get(table, []):
        df[c] = pd.array(np.round(df[c].astype(float)), dtype="Int64")
    return df


def merge_csv(path: Path, table: str, new: pd.DataFrame) -> int:
    """Append rows whose key isn't in `path` yet. Returns the number of rows added."""
    keys = KEYS[table]
    new = _format_for_csv(table, new).drop_duplicates(keys)
    existing = None
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        if list(existing.columns) != SCHEMAS[table]:
            raise ValueError(f"{path} columns {list(existing.columns)} != schema {SCHEMAS[table]}")
        have = set(existing[keys].itertuples(index=False, name=None))
        new = new[[k not in have for k in new[keys].itertuples(index=False, name=None)]]

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if existing is not None:
        existing.to_csv(tmp, index=False, encoding="utf-8")
        new.to_csv(tmp, mode="a", header=False, index=False, encoding="utf-8")
    else:
        new.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    return len(new)


# --------------------------------------------------------------------------- validation

def validate_raw(raw_dir: Path, today: date | None = None) -> dict:
    """Run consistency checks on the raw tables. Returns {'checks': [...], 'shapes': {...}, 'nulls': {...}}."""
    today = today or date.today()
    t = {name: pd.read_csv(raw_dir / f"{name}.csv", dtype={"fight_id": str, "fighter_id": str,
                                                           "event_id": str, "opponent_id": str,
                                                           "fighter_1_id": str, "fighter_2_id": str,
                                                           "winner_id": str})
         for name in SCHEMAS}
    ev, fi, st, fr = t["events"], t["fights"], t["fight_stats"], t["fighters"]
    checks = []

    def check(name, bad: pd.DataFrame | pd.Series, examples_cols=None):
        n = int(bad.sum()) if isinstance(bad, pd.Series) and bad.dtype == bool else len(bad)
        ex = []
        if n:
            if isinstance(bad, pd.Series):
                ex = bad[bad].index[:5].tolist()
            else:
                ex = bad[examples_cols or bad.columns[:3]].head(5).to_dict("records")
        checks.append({"check": name, "violations": n, "examples": ex})

    for name, df in t.items():
        check(f"{name}: duplicate keys", df[df.duplicated(KEYS[name], keep=False)], KEYS[name])

    per_fight = st.groupby("fight_id").size()
    counts = fi["fight_id"].map(per_fight).fillna(0).astype(int)
    check("fights with 0 stat rows (no stats in source)", fi[counts == 0], ["fight_id", "event_date"])
    check("fights with 1 or >2 stat rows", fi[(counts != 0) & (counts != 2)], ["fight_id", "event_date"])

    m = st.merge(fi[["fight_id", "fighter_1_id", "fighter_2_id"]], on="fight_id", how="left")
    check("stat rows whose fighter is not in the fight",
          m[(m.fighter_id != m.fighter_1_id) & (m.fighter_id != m.fighter_2_id)], ["fight_id", "fighter_id"])
    check("stat rows whose fight is not in fights.csv", m[m.fighter_1_id.isna()], ["fight_id", "fighter_id"])

    win = fi.result == "win"
    check("win: winner_id not one of the two fighters",
          fi[win & (fi.winner_id != fi.fighter_1_id) & (fi.winner_id != fi.fighter_2_id)], ["fight_id"])
    check("draw/nc with a winner_id", fi[~win & fi.winner_id.notna()], ["fight_id", "result"])
    check("result not in win/draw/nc", fi[~fi.result.isin(["win", "draw", "nc"])], ["fight_id", "result"])

    for s in ("sig_str", "total_str", "td"):
        check(f"{s}: landed > attempted", st[st[f"{s}_landed"] > st[f"{s}_attempted"]], ["fight_id", "fighter_id"])
    tgt = st[["sig_head_landed", "sig_body_landed", "sig_leg_landed"]].sum(axis=1, min_count=3)
    check("head+body+leg != sig_str_landed",
          st[tgt.notna() & st.sig_str_landed.notna() & (tgt != st.sig_str_landed)], ["fight_id", "fighter_id"])
    pos = st[["sig_distance_landed", "sig_clinch_landed", "sig_ground_landed"]].sum(axis=1, min_count=3)
    check("distance+clinch+ground != sig_str_landed",
          st[pos.notna() & st.sig_str_landed.notna() & (pos != st.sig_str_landed)], ["fight_id", "fighter_id"])
    check("sig_str_landed > total_str_landed",
          st[st.sig_str_landed > st.total_str_landed], ["fight_id", "fighter_id"])
    check("ctrl_sec > fight_duration_sec",
          st[st.ctrl_sec > st.fight_duration_sec], ["fight_id", "fighter_id", "ctrl_sec", "fight_duration_sec"])
    check("end_round > scheduled_rounds",
          fi[fi.end_round > fi.scheduled_rounds], ["fight_id", "end_round", "scheduled_rounds"])

    check("events dated in the future", ev[ev.event_date > today.isoformat()], ["event_id", "event_date"])
    check("fights whose event is not in events.csv", fi[~fi.event_id.isin(ev.event_id)], ["fight_id", "event_id"])
    check("fights.event_date != events.event_date",
          fi[fi.event_date != fi.event_id.map(dict(zip(ev.event_id, ev.event_date)))], ["fight_id"])
    known = set(fr.fighter_id)
    check("fight fighter ids not in fighters.csv",
          fi[~fi.fighter_1_id.isin(known) | ~fi.fighter_2_id.isin(known)], ["fight_id"])
    check("events with no fights", ev[~ev.event_id.isin(fi.event_id)], ["event_id", "event_name"])

    return {
        "checks": checks,
        "shapes": {k: v.shape for k, v in t.items()},
        "nulls": {k: v.isna().sum()[lambda s: s > 0].to_dict() for k, v in t.items()},
    }


def log_validation(report: dict) -> int:
    for name, shape in report["shapes"].items():
        logger.info("%-12s rows=%d cols=%d nulls=%s", name, shape[0], shape[1], report["nulls"][name])
    failed = 0
    for c in report["checks"]:
        level = logging.INFO if c["violations"] == 0 else logging.WARNING
        logger.log(level, "[%s] %s: %d%s", "ok" if c["violations"] == 0 else "!!", c["check"],
                   c["violations"], f" e.g. {c['examples']}" if c["examples"] else "")
        failed += c["violations"] > 0
    return failed


# --------------------------------------------------------------------------- entry point

def build_tables(src: dict[str, pd.DataFrame], overrides: dict[tuple[str, str], str],
                 today: date, scraped_at: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Map the source frames to the four raw tables. Returns (tables, dropped fights)."""
    fighters, name_pairs = build_fighters(src, scraped_at)
    matcher = FighterMatcher(name_pairs, fighter_info(fighters))
    events = build_events(src, scraped_at, today)
    fights, name_map, dropped, bout_keys = build_fights(src, events, matcher, overrides)
    stats = build_fight_stats(src, fights, name_map, bout_keys)
    events = events[events["event_id"].isin(set(fights["event_id"]))]
    return {"fighters": fighters, "fights": fights, "fight_stats": stats, "events": events}, dropped


def run_ingest(full: bool = False, client: HttpClient | None = None, today: date | None = None) -> dict:
    cfg = load_config()
    client = client or get_client()
    today = today or date.today()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    files = cfg["source"]["files"]
    scraped_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    sha, src_dir = fetch_source(client, cfg, offline=full)
    src = load_source(src_dir, files)
    logger.info("Loaded source %s: %s", sha[:10], {k: len(v) for k, v in src.items()})

    overrides = load_overrides(resolve_path(cfg["source"]["overrides_file"]))
    tables, dropped = build_tables(src, overrides, today, scraped_at)
    if full:
        for name in WRITE_ORDER:
            (raw_dir / f"{name}.csv").unlink(missing_ok=True)
    added = {name: merge_csv(raw_dir / f"{name}.csv", name, tables[name]) for name in WRITE_ORDER}

    dropped.to_csv(raw_dir / "ingest_dropped.csv", index=False, encoding="utf-8")
    manifest = {"source_repo": cfg["source"]["repo"], "commit_sha": sha, "ingested_at": scraped_at,
                "source_rows": {k: len(v) for k, v in src.items()},
                "built_rows": {k: len(v) for k, v in tables.items()},
                "added_rows": added, "dropped_fights": len(dropped)}
    manifest_path = resolve_path(cfg["source"]["source_dir"]) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Added rows: %s", added)
    logger.info("Dropped fights: %d (see %s)", len(dropped), raw_dir / "ingest_dropped.csv")
    logger.info("HTTP: %s", client.stats)
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="rebuild raw tables from the newest cached source")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        if not args.validate_only:
            run_ingest(full=args.full)
        raw_dir = resolve_path(load_config()["paths"]["raw_dir"])
        failed = log_validation(validate_raw(raw_dir))
        logger.info("Validation: %d check(s) with violations", failed)
    except Exception:
        logger.exception("Ingest failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
