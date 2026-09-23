import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean import SCHEMA, clean_tables
from src.config import load_config, resolve_path
from src.features import (FIGHTER_FEATURES, RATES, assign_split, build_history, build_training_features,
                          matchup_features, symmetric_probability)

TS = "2026-09-23T00:00:00+00:00"
SPLIT = {"train_end": "2009-01-01", "valid_end": "2010-01-01"}


# --------------------------------------------------------------------------- synthetic data

def _fight(fid, date, f1, f2, result="win", winner="f1", method="KO/TKO", bout_order="1", wc="Lightweight"):
    return {"fight_id": fid, "event_id": f"e{date}", "event_date": date, "bout_order": bout_order,
            "fighter_1_id": f1, "fighter_2_id": f2, "fighter_1_name": f1, "fighter_2_name": f2,
            "winner_id": {"f1": f1, "f2": f2}.get(winner) if result == "win" else None,
            "result": result, "method": method, "end_round": "1", "end_time_sec": "100",
            "scheduled_rounds": "3", "weight_class": wc, "is_title_fight": "0"}


def _stat(fid, me, opp, sl, sa, dur, td=(0, 0), kd=0, ctrl=60, sub=0):
    row = {c: "0" for c in SCHEMA["fight_stats"]}
    row.update({"fight_id": fid, "fighter_id": me, "opponent_id": opp, "sig_str_landed": str(sl),
                "sig_str_attempted": str(sa), "total_str_landed": str(sl), "total_str_attempted": str(sa),
                "td_landed": str(td[0]), "td_attempted": str(td[1]), "kd": str(kd), "ctrl_sec": str(ctrl),
                "sub_att": str(sub), "sig_head_landed": str(sl), "sig_distance_landed": str(sl),
                "fight_duration_sec": str(dur)})
    return row


def synthetic_tables():
    fights = [
        # pre-cutoff (history only): a wins twice on the same night (tournament), b beats c
        _fight("p1", "2003-01-01", "a", "b", bout_order="2"),
        _fight("p2", "2003-01-01", "a", "c", bout_order="1", method="Submission"),
        _fight("p3", "2004-01-01", "b", "c"),
        # post-cutoff
        _fight("f1", "2006-01-01", "a", "b", winner="f2"),                          # b KOs a
        _fight("f2", "2006-01-01", "c", "d", method="Submission"),                  # same card
        _fight("f3", "2007-01-01", "a", "c", result="draw", method="Decision - Split"),
        _fight("f4", "2008-01-01", "a", "d", method="Decision - Unanimous"),
        _fight("f5", "2009-06-01", "b", "e", winner="f2", wc="Welterweight"),        # e debuts
        _fight("f6", "2010-01-01", "a", "e", result="nc", method="Overturned"),
        _fight("f7", "2011-01-01", "e", "a", method="Submission"),
        _fight("f8", "2011-01-01", "b", "d", winner="f2", wc="Welterweight"),
    ]
    stats = [
        _stat("f1", "a", "b", 10, 40, 150, td=(1, 4)), _stat("f1", "b", "a", 20, 30, 150, kd=1),
        _stat("f2", "c", "d", 5, 10, 200, td=(2, 3), sub=2), _stat("f2", "d", "c", 8, 20, 200),
        _stat("f3", "a", "c", 50, 100, 900, td=(2, 2)), _stat("f3", "c", "a", 40, 80, 900, td=(1, 5)),
        _stat("f4", "a", "d", 30, 60, 900), _stat("f4", "d", "a", 25, 70, 900),
        _stat("f5", "b", "e", 12, 24, 300), _stat("f5", "e", "b", 18, 20, 300),
        _stat("f6", "a", "e", 3, 5, 60), _stat("f6", "e", "a", 2, 9, 60),
        _stat("f7", "e", "a", 7, 7, 120), _stat("f7", "a", "e", 1, 3, 120),
        _stat("f8", "b", "d", 9, 9, 900), _stat("f8", "d", "b", 9, 30, 900),
    ]
    events = [{"event_id": f"e{d}", "event_name": d, "event_date": d, "location": "X", "scraped_at": TS}
              for d in sorted({f["event_date"] for f in fights})]
    fighters = [{"fighter_id": i, "name": i, "nickname": None, "height_in": str(68 + k), "weight_lb": "155",
                 "reach_in": str(70 + k), "stance": "Southpaw" if i == "b" else "Orthodox",
                 "dob": f"198{k}-01-01", "scraped_at": TS} for k, i in enumerate("abcde")]
    raw = {"fights": pd.DataFrame(fights), "fight_stats": pd.DataFrame(stats),
           "events": pd.DataFrame(events), "fighters": pd.DataFrame(fighters)}
    tables, _ = clean_tables(raw, "2005-01-01")
    return tables


@pytest.fixture(scope="module")
def tables():
    return synthetic_tables()


@pytest.fixture(scope="module")
def feats(tables):
    return build_training_features(tables, SPLIT)


def row(feats, fid, orientation=0):
    r = feats[(feats.fight_id == fid) & (feats.orientation == orientation)]
    assert len(r) == 1
    return r.iloc[0]


def truncate(tables, date):
    """Only information strictly before `date`."""
    d = pd.Timestamp(date)
    t = dict(tables)
    t["fights"] = tables["fights"][tables["fights"].event_date < d]
    t["fights_prior"] = tables["fights_prior"][tables["fights_prior"].event_date < d]
    t["fight_stats"] = tables["fight_stats"][tables["fight_stats"].fight_id.isin(set(t["fights"].fight_id))]
    return t


def same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if pd.isna(a) and pd.isna(b):
        return True
    return a == pytest.approx(b, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------------- hand-computed values

def test_hand_computed_features(feats):
    r = row(feats, "f4")  # a vs d on 2008-01-01
    # a: 2 prior wins (KO, Sub), loss by KO (2006), draw (2007)
    assert (r.f1_n_fights, r.f1_n_wins, r.f1_n_losses) == (4, 2, 1)
    assert r.f1_win_rate == pytest.approx(3 / 6)
    assert r.f1_streak == 0                     # draw resets the streak
    assert r.f1_finish_rate == 1.0 and r.f1_ko_losses == 1 and r.f1_sub_losses == 0
    assert r.f1_days_since_last_fight == 365
    assert r.f1_n_stat_fights == 2               # prior fights never count for rates
    assert r.f1_sig_landed_p15 == pytest.approx((10 + 50) / (150 + 900) * 900)
    assert r.f1_sig_absorbed_p15 == pytest.approx((20 + 40) / (150 + 900) * 900)
    assert r.f1_sig_acc == pytest.approx(60 / 140)
    assert r.f1_sig_def == pytest.approx(1 - (20 + 40) / (30 + 80))
    assert r.f1_td_def == pytest.approx(1 - 1 / 5)   # f1 opp had 0/0 -> only f3 counts
    assert r.f1_age == pytest.approx((pd.Timestamp("2008-01-01") - pd.Timestamp("1980-01-01")).days / 365.25)
    # d: one fight, a submission loss on 2006-01-01
    assert (r.f2_n_fights, r.f2_n_losses, r.f2_sub_losses, r.f2_streak) == (1, 1, 1, -1)
    assert r.diff_n_fights == 3


def test_last_n_window(feats):
    r = row(feats, "f7", orientation=1)  # a is fighter_1 in the swapped orientation, 2011-01-01
    # a's stat fights before 2011: f1, f3, f4, f6 -> last 3 = f3, f4, f6
    assert r.f1_sig_landed_p15_l3 == pytest.approx((50 + 30 + 3) / (900 + 900 + 60) * 900)
    assert r.f1_sig_landed_p15 == pytest.approx((10 + 50 + 30 + 3) / (150 + 900 + 900 + 60) * 900)


def test_veteran_from_prior_history_is_not_a_debut(feats):
    r = row(feats, "f1")  # a vs b on 2006-01-01, both only have pre-2005 fights
    assert r.f1_is_debut == 0 and r.f1_n_fights == 2 and r.f1_n_wins == 2 and r.f1_streak == 2
    assert r.f2_n_fights == 2 and r.f2_streak == 1   # b: lost to a (2003), beat c (2004)
    for rate in RATES:                               # but no stats-based rates
        assert np.isnan(r[f"f1_{rate}"]) and np.isnan(r[f"f1_{rate}_l3"])


def test_debut_is_nan_not_zero(feats):
    r = row(feats, "f5")  # b vs e, e debuts
    assert r.f2_is_debut == 1 and r.f2_n_fights == 0
    assert np.isnan(r.f2_days_since_last_fight) and np.isnan(r.f2_finish_rate)
    for rate in RATES:
        assert np.isnan(r[f"f2_{rate}"])
    assert r.f2_first_in_weight_class == 1 and r.f1_first_in_weight_class == 1  # b's first welterweight fight
    assert not np.isnan(r.f2_height_in)          # physical data is known for debutants


# --------------------------------------------------------------------------- leakage

def test_no_row_uses_same_day_or_later_history(feats):
    for c in ("_f1_source_max_date", "_f2_source_max_date"):
        used = feats[c].notna()
        assert (feats.loc[used, c] < feats.loc[used, "event_date"]).all()


def test_same_card_fights_are_excluded(tables):
    history = build_history(tables)
    # On 2003-01-01 a fought twice; a fight "on" that date must see none of them.
    pair = pd.DataFrame({"event_date": [pd.Timestamp("2003-01-01"), pd.Timestamp("2003-01-02")],
                         "fighter_1_id": ["a", "a"], "fighter_2_id": ["c", "c"],
                         "weight_class": ["Lightweight"] * 2, "is_title_fight": [0, 0], "scheduled_rounds": [3, 3]})
    f = matchup_features(pair, history)
    assert f.f1_n_fights.tolist() == [0, 2] and f.f1_is_debut.tolist() == [1, 0]


def assert_features_match_truncated_history(tables, feats, fight_ids):
    compare = [c for c in feats.columns if c.startswith(("f1_", "f2_", "diff_"))] + ["stance_matchup"]
    for fid in fight_ids:
        full = feats[feats.fight_id == fid].sort_values("orientation")
        d = full.event_date.iloc[0]
        t = truncate(tables, d)
        pairs = full[["fight_id", "event_date", "fighter_1_id", "fighter_2_id", "weight_class",
                      "is_title_fight", "scheduled_rounds"]]
        again = matchup_features(pairs, build_history(t))
        for c in compare:
            for x, y in zip(full[c].tolist(), again[c].tolist()):
                assert same(x, y), f"{fid} {c}: full={x} truncated={y}"


def test_features_equal_recomputation_from_earlier_fights_only(tables, feats):
    assert_features_match_truncated_history(tables, feats, feats.fight_id.unique())


def test_future_fights_do_not_change_features(tables, feats):
    """Adding later fights must not change any earlier feature row."""
    early = build_training_features(truncate(tables, "2009-01-01"), SPLIT)
    full = feats[feats.fight_id.isin(early.fight_id)].reset_index(drop=True)
    cols = [c for c in early.columns if c.startswith(("f1_", "f2_", "diff_"))]
    pd.testing.assert_frame_equal(early[cols].reset_index(drop=True), full[cols])


REAL = resolve_path(load_config()["paths"]["processed_dir"]) / "fights.csv"


@pytest.mark.skipif(not REAL.exists(), reason="processed data not built")
def test_leakage_on_real_data_sample():
    from src.clean import load_processed
    tables = load_processed()
    feats = build_training_features(tables, load_config()["split"])
    for c in ("_f1_source_max_date", "_f2_source_max_date"):
        used = feats[c].notna()
        assert (feats.loc[used, c] < feats.loc[used, "event_date"]).all()
    rng = np.random.default_rng(0)
    sample = rng.choice(feats.fight_id.unique(), size=12, replace=False)
    assert_features_match_truncated_history(tables, feats, sample)


# --------------------------------------------------------------------------- registry & provenance

def test_every_model_column_is_registered():
    from src.features import CATEGORICAL, FEATURE_CATEGORIES, feature_category, numeric_feature_columns
    cats = {c: feature_category(c) for c in numeric_feature_columns() + CATEGORICAL}
    assert set(cats.values()) == set(FEATURE_CATEGORIES)
    assert cats["diff_win_rate"] == "historical" and cats["f1_sig_def_l3"] == "historical"
    assert cats["f2_reach_in"] == "imputed" and cats["diff_age"] == "imputed"
    assert cats["f1_stance"] == "static" and cats["weight_class"] == "contextual"
    with pytest.raises(KeyError):
        feature_category("f1_betting_odds")


def test_provenance_invariant_raises_value_error(feats):
    from src.features import check_provenance
    check_provenance(feats)                                   # real rows pass
    bad = feats.copy()
    bad.loc[5, "_f1_source_max_date"] = bad.loc[5, "event_date"]   # source on the fight date
    with pytest.raises(ValueError, match="provenance"):
        check_provenance(bad)
    bad = feats.copy()
    bad.loc[3, "_f2_source_max_date"] = bad.loc[3, "event_date"] + pd.Timedelta(days=30)
    with pytest.raises(ValueError, match="provenance"):
        check_provenance(bad)


def test_source_max_date_covers_weight_class_history(tables):
    # b's welterweight history (f5, 2009-06-01) feeds first_in_weight_class of f8 (2011-01-01).
    f = build_training_features(tables, SPLIT)
    r = row(f, "f8")
    assert r.f1_first_in_weight_class == 0
    assert r._f1_source_max_date == pd.Timestamp("2009-06-01")


def test_audit_mode_lists_only_earlier_source_fights(tables):
    f = build_training_features(tables, SPLIT, audit=True)
    r = row(f, "f4")                   # a vs d on 2008-01-01
    assert r._f1_source_fight_ids.split(";") == ["p1", "p2", "f1", "f3"]
    assert r._f2_source_fight_ids == "f2"
    r = row(f, "f5")                   # e debuts
    assert r._f2_source_fight_ids == ""
    fights = pd.concat([tables["fights"], tables["fights_prior"]]).set_index("fight_id").event_date
    for _, x in f.iterrows():
        for ids in (x._f1_source_fight_ids, x._f2_source_fight_ids):
            assert all(fights[i] < x.event_date for i in ids.split(";") if i)
    assert "_f1_source_fight_ids" not in build_training_features(tables, SPLIT).columns   # off by default


# --------------------------------------------------------------------------- symmetry & split

def test_both_orientations_present_and_mirrored(feats):
    assert (feats.groupby("fight_id").size() == 2).all()
    for fid in feats.fight_id.unique():
        a, b = row(feats, fid, 0), row(feats, fid, 1)
        assert (a.fighter_1_id, a.fighter_2_id) == (b.fighter_2_id, b.fighter_1_id)
        assert a.target + b.target == 1
        assert a.split == b.split and a.event_date == b.event_date
        for f in FIGHTER_FEATURES:
            assert same(a[f"f1_{f}"], b[f"f2_{f}"]) and same(a[f"f2_{f}"], b[f"f1_{f}"])
            assert same(a[f"diff_{f}"], -b[f"diff_{f}"]) or (pd.isna(a[f"diff_{f}"]) and pd.isna(b[f"diff_{f}"]))


def test_targets_only_for_fights_with_a_winner(feats, tables):
    assert set(feats.fight_id) == set(tables["fights"].query("result == 'win'").fight_id)
    assert "f3" not in set(feats.fight_id) and "f6" not in set(feats.fight_id)
    assert feats.groupby("orientation").target.mean().tolist() == [
        feats[feats.orientation == 0].target.mean(), 1 - feats[feats.orientation == 0].target.mean()]


def test_split_is_time_based_and_shared(feats):
    assert set(feats.split) == {"train", "valid", "test"}
    assert (feats[feats.split == "train"].event_date < pd.Timestamp(SPLIT["train_end"])).all()
    assert (feats[feats.split == "test"].event_date >= pd.Timestamp(SPLIT["valid_end"])).all()
    assert (feats.groupby("fight_id").split.nunique() == 1).all()
    s = assign_split(pd.Series(pd.to_datetime(["2008-12-31", "2009-01-01", "2010-01-01"])), SPLIT)
    assert s.tolist() == ["train", "valid", "test"]


def test_symmetric_probability_sums_to_one():
    rng = np.random.default_rng(1)
    p_ab, p_ba = rng.uniform(size=100), rng.uniform(size=100)
    assert np.allclose(symmetric_probability(p_ab, p_ba) + symmetric_probability(p_ba, p_ab), 1.0)


def test_symmetry_with_a_fitted_model(feats):
    from sklearn.linear_model import LogisticRegression
    cols = ["diff_n_fights", "diff_win_rate", "diff_streak", "diff_height_in", "diff_reach_in"]
    X = feats[cols].fillna(0).to_numpy()
    model = LogisticRegression().fit(X, feats.target)
    o0, o1 = feats[feats.orientation == 0], feats[feats.orientation == 1].set_index("fight_id").loc[
        feats[feats.orientation == 0].fight_id].reset_index()
    p_ab = model.predict_proba(o0[cols].fillna(0).to_numpy())[:, 1]
    p_ba = model.predict_proba(o1[cols].fillna(0).to_numpy())[:, 1]
    p_a = symmetric_probability(p_ab, p_ba)
    p_b = symmetric_probability(p_ba, p_ab)
    assert np.allclose(p_a + p_b, 1.0)


# --------------------------------------------------------------------------- prediction path

def test_unknown_fighter_is_debut(tables):
    pair = pd.DataFrame({"event_date": [pd.Timestamp("2030-01-01")], "fighter_1_id": ["a"],
                         "fighter_2_id": [None], "weight_class": ["Lightweight"], "is_title_fight": [0],
                         "scheduled_rounds": [3]})
    f = matchup_features(pair, build_history(tables)).iloc[0]
    assert f.f2_is_debut == 1 and f.f2_n_fights == 0 and np.isnan(f.f2_height_in)
    assert f.f1_n_fights == 7 and f.f1_is_debut == 0  # p1, p2, f1, f3, f4, f6, f7
