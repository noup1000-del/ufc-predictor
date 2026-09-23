"""Stage 3: pre-fight features, one row per fight orientation.

No leakage by construction: every fighter's history is collapsed to one cumulative
state per (fighter, event_date), and a fight on date D reads the latest state with
hist_date < D (`merge_asof(..., allow_exact_matches=False)`), so fights on the same
date -- including the fight itself and the rest of its card -- are never used.

Training (`build_training_features`) and prediction (`matchup_features` on an
upcoming card) use the same `matchup_features` code path.

Usage: python -m src.features
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import load_processed
from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

UNKNOWN = "__unknown__"
SECONDS_15_MIN = 900
LAST_N = 3

# name -> (numerator column, denominator column, scale). Rates use only fights with stats
# (post-cutoff); a fight contributes to a rate only if both its numerator and denominator exist.
RATES = {
    "sig_landed_p15": ("sig_str_landed", "fight_duration_sec", SECONDS_15_MIN),
    "sig_absorbed_p15": ("opp_sig_str_landed", "fight_duration_sec", SECONDS_15_MIN),
    "sig_acc": ("sig_str_landed", "sig_str_attempted", 1),
    "sig_def": ("opp_sig_str_missed", "opp_sig_str_attempted", 1),
    "kd_p15": ("kd", "fight_duration_sec", SECONDS_15_MIN),
    "td_landed_p15": ("td_landed", "fight_duration_sec", SECONDS_15_MIN),
    "td_acc": ("td_landed", "td_attempted", 1),
    "td_def": ("opp_td_missed", "opp_td_attempted", 1),
    "sub_att_p15": ("sub_att", "fight_duration_sec", SECONDS_15_MIN),
    "ctrl_p15": ("ctrl_sec", "fight_duration_sec", SECONDS_15_MIN),
}
COUNTS = ["fights", "wins", "losses", "draws", "ncs", "ko_wins", "sub_wins", "ko_losses", "sub_losses",
          "stat_fights"]

FIGHTER_FEATURES = (
    ["n_fights", "n_wins", "n_losses", "win_rate", "streak", "days_since_last_fight", "is_debut",
     "finish_rate", "ko_losses", "sub_losses", "n_stat_fights"]
    + list(RATES)
    + [f"{r}_l{LAST_N}" for r in RATES]
    + ["height_in", "reach_in", "age", "first_in_weight_class"]
)
CONTEXT_NUMERIC = ["is_title_fight", "scheduled_rounds"]
CATEGORICAL = ["weight_class", "stance_matchup", "f1_stance", "f2_stance"]
ID_COLUMNS = ["fight_id", "event_date", "fighter_1_id", "fighter_2_id", "orientation", "target", "split"]


def numeric_feature_columns() -> list[str]:
    return (CONTEXT_NUMERIC + [f"f1_{f}" for f in FIGHTER_FEATURES] + [f"f2_{f}" for f in FIGHTER_FEATURES]
            + [f"diff_{f}" for f in FIGHTER_FEATURES])


# --------------------------------------------------------------------------- feature registry

HISTORICAL, STATIC, CONTEXTUAL, IMPUTED = "historical", "static", "contextual", "imputed"
FEATURE_CATEGORIES = (HISTORICAL, STATIC, CONTEXTUAL, IMPUTED)
# historical: derived from the fighter's earlier fights (must satisfy the provenance invariant)
# static:     fighter attribute that does not depend on fights (bio)
# contextual: known property of the bout itself, fixed before the fight
# imputed:    bio-derived value that train.PhysicalImputer fills when missing (fit on training rows only)
FEATURE_REGISTRY: dict[str, str] = {
    **{f: HISTORICAL for f in ["n_fights", "n_wins", "n_losses", "win_rate", "streak", "days_since_last_fight",
                               "is_debut", "finish_rate", "ko_losses", "sub_losses", "n_stat_fights",
                               "first_in_weight_class"]},
    **{r: HISTORICAL for r in RATES},
    **{f"{r}_l{LAST_N}": HISTORICAL for r in RATES},
    "height_in": IMPUTED, "reach_in": IMPUTED, "age": IMPUTED,
    "stance": STATIC,
    "weight_class": CONTEXTUAL, "is_title_fight": CONTEXTUAL, "scheduled_rounds": CONTEXTUAL,
    "stance_matchup": CONTEXTUAL,
}


def feature_category(column: str) -> str:
    """Category of a model column ('f1_age' -> imputed, 'diff_win_rate' -> historical, ...)."""
    base = column
    for prefix in ("f1_", "f2_", "diff_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    if base not in FEATURE_REGISTRY:
        raise KeyError(f"feature {column!r} is not in FEATURE_REGISTRY")
    return FEATURE_REGISTRY[base]


def check_provenance(feats: pd.DataFrame) -> None:
    """Hard invariant: for every row with history, max(source event_date) < fight date.
    Uses the `_f1_source_max_date` / `_f2_source_max_date` columns from matchup_features."""
    for side in ("1", "2"):
        src = pd.to_datetime(feats[f"_f{side}_source_max_date"])
        bad = src.notna() & (src >= pd.to_datetime(feats["event_date"]))
        if bad.any():
            ex = feats.loc[bad, ["event_date", f"fighter_{side}_id"]].head(3).to_dict("records")
            raise ValueError(f"provenance violation: fighter_{side} history dated on/after the fight: {ex}")


# --------------------------------------------------------------------------- history

@dataclass
class History:
    states: pd.DataFrame        # one row per (fighter_id, hist_date): cumulative state after that date
    weight_classes: pd.DataFrame  # (fighter_id, weight_class, hist_date) of every past fight
    fighters: pd.DataFrame      # fighter_id -> height_in, reach_in, stance, dob
    sources: pd.DataFrame | None = None  # (fighter_id, event_date, fight_id) of every appearance, for audits


def appearances(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per fighter per fight (post-cutoff fights + history-only prior fights)."""
    cols = ["fight_id", "event_date", "bout_order", "fighter_1_id", "fighter_2_id", "winner_id", "result",
            "method_group", "weight_class", "has_stats"]
    fights = pd.concat([tables["fights"][cols].assign(is_prior=False),
                        tables["fights_prior"][cols].assign(is_prior=True)], ignore_index=True)
    fights["weight_class"] = fights["weight_class"].astype("string").fillna(UNKNOWN)
    sides = []
    for me, opp in (("fighter_1_id", "fighter_2_id"), ("fighter_2_id", "fighter_1_id")):
        s = fights.rename(columns={me: "fighter_id", opp: "opponent_id"})
        sides.append(s.drop(columns=[c for c in ("fighter_1_id", "fighter_2_id") if c in s]))
    a = pd.concat(sides, ignore_index=True)

    res = a["result"].astype("string").fillna("nc")
    won = (res == "win") & (a["winner_id"] == a["fighter_id"])
    lost = (res == "win") & (a["winner_id"] != a["fighter_id"])
    mg = a["method_group"].astype("string")
    a["outcome"] = np.select([won.fillna(False), lost.fillna(False), res == "draw"], ["win", "loss", "draw"], "nc")
    a["fights"] = 1
    a["wins"] = (a["outcome"] == "win").astype(int)
    a["losses"] = (a["outcome"] == "loss").astype(int)
    a["draws"] = (a["outcome"] == "draw").astype(int)
    a["ncs"] = (a["outcome"] == "nc").astype(int)
    a["ko_wins"] = (a["wins"].astype(bool) & (mg == "ko_tko").fillna(False)).astype(int)
    a["sub_wins"] = (a["wins"].astype(bool) & (mg == "submission").fillna(False)).astype(int)
    a["ko_losses"] = (a["losses"].astype(bool) & (mg == "ko_tko").fillna(False)).astype(int)
    a["sub_losses"] = (a["losses"].astype(bool) & (mg == "submission").fillna(False)).astype(int)

    st = tables["fight_stats"]
    own = st[["fight_id", "fighter_id", "sig_str_landed", "sig_str_attempted", "kd", "td_landed",
              "td_attempted", "sub_att", "ctrl_sec", "fight_duration_sec"]]
    opp = st[["fight_id", "fighter_id", "sig_str_landed", "sig_str_attempted", "td_landed", "td_attempted"]]
    opp = opp.rename(columns={"fighter_id": "opponent_id", "sig_str_landed": "opp_sig_str_landed",
                              "sig_str_attempted": "opp_sig_str_attempted", "td_landed": "opp_td_landed",
                              "td_attempted": "opp_td_attempted"})
    a = a.merge(own, on=["fight_id", "fighter_id"], how="left").merge(opp, on=["fight_id", "opponent_id"], how="left")
    stat_cols = [c for c in a.columns if c in own.columns or c in opp.columns]
    stat_cols = [c for c in stat_cols if c not in ("fight_id", "fighter_id", "opponent_id")]
    for c in stat_cols:
        a[c] = a[c].astype("float64")
    # Prior (pre-cutoff) fights never feed rates, even if stats were present.
    usable = ~a["is_prior"] & a["has_stats"].fillna(False).astype(bool) & a["sig_str_landed"].notna()
    a.loc[~usable, stat_cols] = np.nan
    a["stat_fights"] = usable.astype(int)
    a["opp_sig_str_missed"] = a["opp_sig_str_attempted"] - a["opp_sig_str_landed"]
    a["opp_td_missed"] = a["opp_td_attempted"] - a["opp_td_landed"]

    # Tie-break within a date (pre-2001 tournaments): ufcstats lists later bouts first.
    a = a.sort_values(["fighter_id", "event_date", "bout_order", "fight_id"],
                      ascending=[True, True, False, True]).reset_index(drop=True)
    return a


def _streaks(a: pd.DataFrame) -> np.ndarray:
    """Win streak (+n) / loss streak (-n) after each appearance; draws reset, NCs don't count."""
    out = np.zeros(len(a), dtype=int)
    prev_fighter, streak = None, 0
    for i, (fid, outcome) in enumerate(zip(a["fighter_id"].to_numpy(), a["outcome"].to_numpy())):
        if fid != prev_fighter:
            prev_fighter, streak = fid, 0
        if outcome == "win":
            streak = streak + 1 if streak > 0 else 1
        elif outcome == "loss":
            streak = streak - 1 if streak < 0 else -1
        elif outcome == "draw":
            streak = 0
        out[i] = streak
    return out


def build_history(tables: dict[str, pd.DataFrame]) -> History:
    a = appearances(tables)
    g = a.groupby("fighter_id", sort=False)

    state = a[["fighter_id", "event_date"]].copy()
    for c in COUNTS:
        state[f"cum_{c}"] = g[c].cumsum()
    state["streak"] = _streaks(a)

    stat_rows = a["stat_fights"] == 1
    for name, (num, den, _) in RATES.items():
        both = a[num].notna() & a[den].notna() & stat_rows
        n = a[num].where(both, 0.0)
        d = a[den].where(both, 0.0)
        state[f"c_num_{name}"] = n.groupby(a["fighter_id"]).cumsum()
        state[f"c_den_{name}"] = d.groupby(a["fighter_id"]).cumsum()
        # Last-N window over fights with stats only, carried forward over no-stats fights.
        sub = pd.DataFrame({"f": a.loc[stat_rows, "fighter_id"], "n": n[stat_rows], "d": d[stat_rows]})
        roll = sub.groupby("f")[["n", "d"]].rolling(LAST_N, min_periods=1).sum().reset_index(level=0, drop=True)
        state[f"l_num_{name}"] = roll["n"].reindex(a.index)
        state[f"l_den_{name}"] = roll["d"].reindex(a.index)
        for c in (f"l_num_{name}", f"l_den_{name}"):
            state[c] = state.groupby("fighter_id", sort=False)[c].ffill()

    # State after the last fight of each date.
    states = state.groupby(["fighter_id", "event_date"], sort=False).tail(1)
    states = states.rename(columns={"event_date": "hist_date"}).reset_index(drop=True)

    wc = a[["fighter_id", "weight_class", "event_date"]].drop_duplicates()
    wc = wc.rename(columns={"event_date": "hist_date"}).reset_index(drop=True)

    fighters = tables["fighters"][["fighter_id", "height_in", "reach_in", "stance", "dob"]].copy()
    sources = a[["fighter_id", "event_date", "fight_id"]].sort_values(["fighter_id", "event_date"]).reset_index(drop=True)
    return History(states=states, weight_classes=wc, fighters=fighters, sources=sources)


def _source_fights(history: History, fighter_ids, dates) -> tuple[list[str], list]:
    """Audit helper: ';'-joined fight_ids (and their max date) a fighter had strictly before each date."""
    src = history.sources.copy()
    src["fighter_id"] = src["fighter_id"].astype(str)
    groups = {fid: (g["event_date"].to_numpy(dtype="datetime64[us]"), g["fight_id"].astype(str).to_numpy())
              for fid, g in src.groupby("fighter_id", sort=False)}
    ids_out, max_out = [], []
    for fid, d in zip(fighter_ids, dates):
        dates_arr, ids = groups.get(str(fid), (np.array([], dtype="datetime64[us]"), np.array([], dtype=str)))
        n = np.searchsorted(dates_arr, np.datetime64(pd.Timestamp(d), "us"), side="left")  # strictly before
        ids_out.append(";".join(ids[:n]))
        max_out.append(pd.Timestamp(dates_arr[n - 1]) if n else pd.NaT)
    return ids_out, max_out


# --------------------------------------------------------------------------- features

def _safe_div(n, d):
    n = pd.to_numeric(n, errors="coerce").astype("float64")
    d = pd.to_numeric(d, errors="coerce").astype("float64")
    return (n / d.where(d > 0)).astype("float64")


def fighter_features(sides: pd.DataFrame, history: History, audit: bool = False) -> pd.DataFrame:
    """Pre-fight features for rows (row_id, fighter_id, event_date, weight_class).
    With audit=True, also returns `_source_fight_ids` (every earlier fight of the fighter)."""
    left = sides.copy()
    left["fighter_id"] = left["fighter_id"].astype("string").fillna(UNKNOWN)
    left["weight_class"] = left["weight_class"].astype("string").fillna(UNKNOWN)
    left["event_date"] = pd.to_datetime(left["event_date"]).astype("datetime64[us]")
    left = left.sort_values("event_date", kind="stable")

    states = history.states.copy()
    states["fighter_id"] = states["fighter_id"].astype("string")
    states["hist_date"] = states["hist_date"].astype("datetime64[us]")
    st = pd.merge_asof(left, states.sort_values("hist_date"), left_on="event_date", right_on="hist_date",
                       by="fighter_id", allow_exact_matches=False, direction="backward")
    # Runtime leakage guard: no state may come from the fight date or later.
    if (st["hist_date"] >= st["event_date"]).any():
        raise ValueError("provenance violation: history state dated on/after the fight date")

    out = pd.DataFrame({"row_id": st["row_id"].to_numpy()})
    fights = st["cum_fights"].fillna(0)
    wins = st["cum_wins"].fillna(0)
    out["n_fights"] = fights.to_numpy()
    out["n_wins"] = wins.to_numpy()
    out["n_losses"] = st["cum_losses"].fillna(0).to_numpy()
    out["win_rate"] = ((wins + 1) / (fights + 2)).to_numpy()
    out["streak"] = st["streak"].fillna(0).to_numpy()
    out["days_since_last_fight"] = (st["event_date"] - st["hist_date"]).dt.days.astype("float64").to_numpy()
    out["is_debut"] = (fights == 0).astype(int).to_numpy()
    out["finish_rate"] = _safe_div(st["cum_ko_wins"].fillna(0) + st["cum_sub_wins"].fillna(0), wins).to_numpy()
    out["ko_losses"] = st["cum_ko_losses"].fillna(0).to_numpy()
    out["sub_losses"] = st["cum_sub_losses"].fillna(0).to_numpy()
    out["n_stat_fights"] = st["cum_stat_fights"].fillna(0).to_numpy()
    for name, (_, _, scale) in RATES.items():
        out[name] = (_safe_div(st[f"c_num_{name}"], st[f"c_den_{name}"]) * scale).to_numpy()
        out[f"{name}_l{LAST_N}"] = (_safe_div(st[f"l_num_{name}"], st[f"l_den_{name}"]) * scale).to_numpy()
    # A "defense" rate means 1 - opponent accuracy; it was built from missed/attempted already.

    fr = history.fighters.copy()
    fr["fighter_id"] = fr["fighter_id"].astype("string")
    phys = st[["fighter_id", "event_date"]].merge(fr, on="fighter_id", how="left")
    out["height_in"] = phys["height_in"].astype("float64").to_numpy()
    out["reach_in"] = phys["reach_in"].astype("float64").to_numpy()
    out["age"] = ((phys["event_date"] - phys["dob"]).dt.days / 365.25).astype("float64").to_numpy()
    out["stance"] = phys["stance"].astype("string").to_numpy()

    wc = history.weight_classes.copy()
    wc["fighter_id"] = wc["fighter_id"].astype("string")
    wc["weight_class"] = wc["weight_class"].astype("string")
    wc["hist_date"] = wc["hist_date"].astype("datetime64[us]")
    wcm = pd.merge_asof(st[["row_id", "fighter_id", "weight_class", "event_date"]],
                        wc.sort_values("hist_date"), left_on="event_date", right_on="hist_date",
                        by=["fighter_id", "weight_class"], allow_exact_matches=False, direction="backward")
    out["first_in_weight_class"] = wcm["hist_date"].isna().astype(int).to_numpy()
    # Latest event any historical feature of this row was computed from.
    out["_source_max_date"] = pd.concat([st["hist_date"], wcm["hist_date"]], axis=1).max(axis=1).to_numpy()
    if audit:
        ids, max_dates = _source_fights(history, st["fighter_id"], st["event_date"])
        out["_source_fight_ids"] = ids
        # Independent check: the audited fight list must end strictly before the fight too.
        audited = pd.Series(max_dates, dtype="datetime64[us]")
        if (audited.notna() & (audited.to_numpy() >= st["event_date"].to_numpy())).any():
            raise ValueError("provenance violation: audited source fight dated on/after the fight date")
    return out


def matchup_features(pairs: pd.DataFrame, history: History, audit: bool = False) -> pd.DataFrame:
    """Features for fights given as (event_date, fighter_1_id, fighter_2_id, weight_class,
    is_title_fight, scheduled_rounds[, fight_id, ...]). Returns one row per input row, same order.
    Always checks the provenance invariant (ValueError on violation). audit=True adds
    `_f1_source_fight_ids` / `_f2_source_fight_ids` (';'-joined earlier fight_ids)."""
    pairs = pairs.reset_index(drop=True)
    sides = pd.concat([
        pd.DataFrame({"row_id": pairs.index, "side": side, "fighter_id": pairs[f"fighter_{side}_id"],
                      "event_date": pairs["event_date"], "weight_class": pairs["weight_class"]})
        for side in ("1", "2")], ignore_index=True)
    sides["row_id"] = sides["row_id"].astype(str) + "_" + sides["side"]
    ff = fighter_features(sides.drop(columns="side"), history, audit=audit).set_index("row_id")

    cols: dict[str, pd.Series] = {}
    parts = {side: ff.loc[[f"{i}_{side}" for i in pairs.index]].reset_index(drop=True) for side in ("1", "2")}
    for side, part in parts.items():
        for f in FIGHTER_FEATURES:
            cols[f"f{side}_{f}"] = part[f]
        cols[f"f{side}_stance"] = part["stance"]
        cols[f"_f{side}_source_max_date"] = part["_source_max_date"]
        if audit:
            cols[f"_f{side}_source_fight_ids"] = part["_source_fight_ids"]
    for f in FIGHTER_FEATURES:
        cols[f"diff_{f}"] = parts["1"][f] - parts["2"][f]
    s1 = parts["1"]["stance"].fillna("unknown").str.lower()
    s2 = parts["2"]["stance"].fillna("unknown").str.lower()
    cols["stance_matchup"] = s1 + "_vs_" + s2
    base = pairs.drop(columns=["is_title_fight", "scheduled_rounds"])
    cols["is_title_fight"] = pairs["is_title_fight"].astype("float64")
    cols["scheduled_rounds"] = pairs["scheduled_rounds"].astype("float64")
    out = pd.concat([base, pd.DataFrame(cols)], axis=1)
    check_provenance(out)
    return out


def swap_orientation(pairs: pd.DataFrame) -> pd.DataFrame:
    s = pairs.copy()
    s["fighter_1_id"], s["fighter_2_id"] = pairs["fighter_2_id"], pairs["fighter_1_id"]
    return s


def assign_split(event_date: pd.Series, split_cfg: dict) -> pd.Series:
    """Time-based split. Both orientations of a fight share event_date, hence the split."""
    d = pd.to_datetime(event_date)
    train_end, valid_end = pd.Timestamp(split_cfg["train_end"]), pd.Timestamp(split_cfg["valid_end"])
    return pd.Series(np.where(d < train_end, "train", np.where(d < valid_end, "valid", "test")),
                     index=event_date.index)


def symmetric_probability(p_ab, p_ba):
    """P(A beats B) from both orientations: (p(A,B) + 1 - p(B,A)) / 2."""
    return (np.asarray(p_ab, dtype=float) + 1.0 - np.asarray(p_ba, dtype=float)) / 2.0


AUDIT_COLUMNS = ["_f1_source_max_date", "_f2_source_max_date"]


def build_training_features(tables: dict[str, pd.DataFrame], split_cfg: dict,
                            history: History | None = None, audit: bool = False) -> pd.DataFrame:
    """Both orientations of every target fight. Keeps `_f{1,2}_source_max_date`; with
    audit=True also `_f{1,2}_source_fight_ids`."""
    history = history or build_history(tables)
    f = tables["fights"]
    f = f[f["is_target"].fillna(False).astype(bool)]
    base = pd.DataFrame({
        "fight_id": f["fight_id"].to_numpy(), "event_date": f["event_date"].to_numpy(),
        "fighter_1_id": f["fighter_1_id"].to_numpy(), "fighter_2_id": f["fighter_2_id"].to_numpy(),
        "weight_class": f["weight_class"].astype("string").to_numpy(),
        "is_title_fight": f["is_title_fight"].astype("float64").to_numpy(),
        "scheduled_rounds": f["scheduled_rounds"].astype("float64").to_numpy(),
        "winner_id": f["winner_id"].to_numpy(),
    })
    both = pd.concat([base.assign(orientation=0), swap_orientation(base).assign(orientation=1)],
                     ignore_index=True)
    both["target"] = (both["winner_id"] == both["fighter_1_id"]).astype(int)
    feats = matchup_features(both, history, audit=audit)
    feats = feats.assign(split=assign_split(feats["event_date"], split_cfg))
    cols = ID_COLUMNS + CATEGORICAL + numeric_feature_columns() + AUDIT_COLUMNS
    if audit:
        cols += ["_f1_source_fight_ids", "_f2_source_fight_ids"]
    return feats[cols].sort_values(
        ["event_date", "fight_id", "orientation"]).reset_index(drop=True)


# --------------------------------------------------------------------------- entry point

def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unregistered(column: str) -> bool:
    try:
        feature_category(column)
        return False
    except KeyError:
        return True


def run_features() -> pd.DataFrame:
    cfg = load_config()
    tables = load_processed()
    feats = build_training_features(tables, cfg["split"])
    check_provenance(feats)  # hard invariant, raises ValueError (already checked per call; explicit here)
    model_columns = numeric_feature_columns() + CATEGORICAL
    unregistered = [c for c in model_columns if _unregistered(c)]
    if unregistered:
        raise ValueError(f"features missing from FEATURE_REGISTRY: {unregistered}")
    out = feats.drop(columns=AUDIT_COLUMNS)

    out_dir = resolve_path(cfg["paths"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fight_features.csv"
    tmp = path.with_name(path.name + ".tmp")
    out.assign(event_date=out["event_date"].dt.strftime("%Y-%m-%d")).to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)
    (out_dir / "feature_list.json").write_text(json.dumps({
        "numeric": numeric_feature_columns(), "categorical": CATEGORICAL, "id_columns": ID_COLUMNS,
        "fighter_features": FIGHTER_FEATURES,
        "categories": {c: feature_category(c) for c in model_columns},
        "provenance_checked": True,
        "features_sha256": file_sha256(path),
        "data_cutoff": str(out["event_date"].max().date())}, indent=2), encoding="utf-8")

    logger.info("Wrote %s: %d rows (%d fights x 2 orientations), %d columns", path, len(out),
                out["fight_id"].nunique(), out.shape[1])
    logger.info("Rows per split: %s", out.groupby("split").size().to_dict())
    return out


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_features()
    except Exception:
        logger.exception("Features failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
