import pickle

import numpy as np
import pandas as pd
import pytest

from src.features import FIGHTER_FEATURES
from src.train import (BaselineModel, FoldLeakError, LGBMModel, LogisticModel, PhysicalImputer,
                       backtest_summary, brier_decomposition, calibration_errors, calibration_table, evaluate,
                       fight_level, flat_metrics, missingness_report, promotion_gates, run_backtest, score,
                       wilson_interval)


def make_rows(n_fights=400, seed=0):
    """Synthetic both-orientation rows where a latent skill difference drives the outcome."""
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=(n_fights, 2))
    y = (skill[:, 0] - skill[:, 1] + rng.normal(scale=0.5, size=n_fights) > 0).astype(int)
    base = {"fight_id": [f"f{i}" for i in range(n_fights)],
            "event_date": pd.date_range("2010-01-01", periods=n_fights, freq="D"),
            "weight_class": rng.choice(["Lightweight", "Welterweight"], n_fights),
            "is_title_fight": 0.0, "scheduled_rounds": 3.0}
    f1 = {f"f1_{f}": rng.normal(size=n_fights) for f in FIGHTER_FEATURES}
    f2 = {f"f2_{f}": rng.normal(size=n_fights) for f in FIGHTER_FEATURES}
    f1["f1_win_rate"], f2["f2_win_rate"] = skill[:, 0], skill[:, 1]
    f1["f1_n_wins"], f2["f2_n_wins"] = np.round(skill[:, 0] * 3 + 5), np.round(skill[:, 1] * 3 + 5)
    f1["f1_height_in"], f2["f2_height_in"] = rng.normal(70, 3, n_fights), rng.normal(70, 3, n_fights)
    f1["f1_reach_in"], f2["f2_reach_in"] = f1["f1_height_in"] + 2, f2["f2_height_in"] + 2
    f1["f1_age"], f2["f2_age"] = rng.normal(30, 4, n_fights), rng.normal(30, 4, n_fights)
    o0 = pd.DataFrame({**base, **f1, **f2, "orientation": 0, "target": y,
                       "stance_matchup": "orthodox_vs_orthodox", "f1_stance": "Orthodox", "f2_stance": "Orthodox"})
    o1 = o0.copy()
    for f in FIGHTER_FEATURES:
        o1[f"f1_{f}"], o1[f"f2_{f}"] = o0[f"f2_{f}"], o0[f"f1_{f}"]
    o1["orientation"], o1["target"] = 1, 1 - y
    rows = pd.concat([o0, o1], ignore_index=True)
    diffs = pd.DataFrame({f"diff_{f}": rows[f"f1_{f}"] - rows[f"f2_{f}"] for f in FIGHTER_FEATURES})
    return pd.concat([rows, diffs], axis=1)


NUMERIC = ["is_title_fight", "scheduled_rounds"] + [f"{p}_{f}" for p in ("f1", "f2", "diff") for f in FIGHTER_FEATURES]
CATEGORICAL = ["weight_class", "stance_matchup", "f1_stance", "f2_stance"]


def test_fight_level_is_symmetric_and_requires_both_orientations():
    rows = make_rows(10)
    p = np.where(rows.orientation == 0, 0.7, 0.4)  # p(A,B)=0.7, p(B,A)=0.4 -> (0.7+0.6)/2
    fl = fight_level(rows, p)
    assert len(fl) == 10 and np.allclose(fl.p, 0.65)
    with pytest.raises(ValueError):
        fight_level(rows[rows.fight_id != "f0"].iloc[1:], p[1:len(rows) - 1])


def test_evaluate_and_coin_flip_is_deterministic():
    fl = pd.DataFrame({"fight_id": ["a", "b", "c", "d"], "target": [1, 0, 1, 0], "p": [0.9, 0.2, 0.5, 0.5]})
    m1, m2 = evaluate(fl), evaluate(fl)
    assert m1 == m2 and m1["n_fights"] == 4
    assert m1["probabilistic"]["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.25, 0.25]))


def test_calibration_table_is_folded_to_the_favourite():
    fl = pd.DataFrame({"fight_id": list("abcd"), "target": [1, 0, 0, 1], "p": [0.9, 0.1, 0.8, 0.52]})
    t = calibration_table(fl, [0.5, 0.6, 1.0])
    assert t.n.sum() == 4
    top = t.iloc[1]  # favourites at 0.9, 0.9, 0.8: won, won, lost
    assert top.n == 3 and top.actual_win_rate == pytest.approx(2 / 3)


def test_baseline_symmetric_probability():
    rows = make_rows(300)
    m = BaselineModel().fit(rows)
    fl = fight_level(rows, m.predict_proba(rows))
    o0 = rows[rows.orientation == 0].set_index("fight_id").loc[fl.fight_id]
    d = (o0.f1_n_wins - o0.f2_n_wins).to_numpy()
    assert np.allclose(fl.p[d > 0], m.p_favourite_) and np.allclose(fl.p[d == 0], 0.5)
    assert m.p_favourite_ > 0.6  # wins correlate with skill in the synthetic data


def test_physical_imputer_fills_from_training_values_and_recomputes_diffs():
    train = make_rows(200)
    imp = PhysicalImputer().fit(train)
    assert imp.reach_slope_ == pytest.approx(1.0) and imp.reach_intercept_ == pytest.approx(2.0)
    test = make_rows(5, seed=1)
    test.loc[0, "f1_reach_in"] = np.nan                       # height known -> regression
    test.loc[1, ["f1_reach_in", "f1_height_in", "f1_age"]] = np.nan  # -> weight-class medians
    out = imp.transform(test)
    assert out.loc[0, "f1_reach_in"] == pytest.approx(test.loc[0, "f1_height_in"] + 2)
    assert out.loc[1, "f1_height_in"] == pytest.approx(imp.wc_median_["height_in"][test.loc[1, "weight_class"]])
    assert not out[[f"{s}_{p}" for s in ("f1", "f2", "diff") for p in ("height_in", "reach_in", "age")]].isna().any().any()
    assert np.allclose(out.diff_reach_in, out.f1_reach_in - out.f2_reach_in)
    assert np.isnan(test.loc[0, "f1_reach_in"])  # input not modified


def test_missingness_report_flags_future_leaking_nan():
    rows = make_rows(400)
    losers = rows.target == 0
    rows.loc[losers & (rows.index % 3 == 0), "f1_reach_in"] = np.nan
    rep = missingness_report(rows, ["f1_reach_in", "f1_age"])
    r = rep.set_index("feature").loc["f1_reach_in"]
    assert r.target_rate_missing == 0.0 and "f1_age" not in set(rep.feature)


def test_models_learn_are_symmetric_and_pickle(tmp_path):
    train, test = make_rows(600, 0), make_rows(200, 1)
    for m in (LogisticModel([c for c in NUMERIC if c.startswith("diff_")], 1.0, 0).fit(train),
              LGBMModel(NUMERIC, CATEGORICAL, {"num_leaves": 7, "learning_rate": 0.05, "random_state": 0},
                        100).fit(train)):
        fl = fight_level(test, m.predict_proba(test))
        assert evaluate(fl)["accuracy"] > 0.7, m.name
        fl_swapped = fight_level(test.assign(orientation=1 - test.orientation, target=1 - test.target),
                                 m.predict_proba(test))
        assert np.allclose(fl.set_index("fight_id").p + fl_swapped.set_index("fight_id").p, 1.0)
        path = tmp_path / f"{m.name}.pkl"
        path.write_bytes(pickle.dumps(m))
        assert np.allclose(pickle.loads(path.read_bytes()).predict_proba(test), m.predict_proba(test))


# --------------------------------------------------------------------------- metric taxonomy

def test_wilson_interval_known_values():
    lo, hi = wilson_interval([5, 0, 10, 0], [10, 10, 10, 0])
    assert lo[0] == pytest.approx(0.2366, abs=1e-4) and hi[0] == pytest.approx(0.7634, abs=1e-4)
    assert lo[1] == pytest.approx(0.0, abs=1e-12) and hi[1] == pytest.approx(0.2775, abs=1e-4)
    assert lo[2] == pytest.approx(0.7225, abs=1e-4) and hi[2] == pytest.approx(1.0)
    assert np.isnan(lo[3]) and np.isnan(hi[3])


def test_calibration_table_has_wilson_ci_containing_rate():
    rows = make_rows(400)
    fl = fight_level(rows, BaselineModel().fit(rows).predict_proba(rows))
    t = calibration_table(fl)
    filled = t[t.n > 0]
    assert {"lower_ci", "upper_ci"} <= set(t.columns)
    assert ((filled.lower_ci <= filled.actual_win_rate) & (filled.actual_win_rate <= filled.upper_ci)).all()


def test_ece_and_mce():
    t = pd.DataFrame({"n": [10, 30, 0], "mean_predicted": [0.55, 0.7, np.nan], "actual_win_rate": [0.45, 0.75, np.nan]})
    ece, mce = calibration_errors(t)
    assert ece == pytest.approx((10 * 0.10 + 30 * 0.05) / 40) and mce == pytest.approx(0.10)


def test_brier_decomposition_is_exact_for_binned_constant_forecasts():
    # Forecasts constant within each bin -> Brier = reliability - resolution + uncertainty exactly.
    p = np.array([0.52] * 6 + [0.62] * 8 + [0.9] * 6 + [0.3] * 5)   # 0.3 folds to 0.7
    y = np.array([1, 0, 1, 0, 1, 1] + [1, 1, 0, 1, 1, 0, 1, 0] + [1, 1, 1, 1, 1, 0] + [0, 0, 1, 0, 0])
    fl = pd.DataFrame({"fight_id": [f"f{i}" for i in range(len(p))], "target": y, "p": p})
    m, d = evaluate(fl), brier_decomposition(fl)
    assert m["probabilistic"]["brier"] == pytest.approx(d["reliability"] - d["resolution"] + d["uncertainty"])
    assert m["calibration"]["brier_reliability"] == pytest.approx(d["reliability"])
    assert m["discrimination"]["brier_resolution"] == pytest.approx(d["resolution"])


def test_metrics_do_not_depend_on_fighter_order():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.8, 300)
    y = (rng.uniform(size=300) < p).astype(int)
    fl = pd.DataFrame({"fight_id": [f"f{i}" for i in range(300)], "target": y, "p": p})
    flip = rng.uniform(size=300) < 0.5
    fl2 = fl.assign(target=np.where(flip, 1 - y, y), p=np.where(flip, 1 - p, p))
    a, b = flat_metrics(evaluate(fl)), flat_metrics(evaluate(fl2))
    for k in ("brier", "log_loss", "roc_auc", "ece", "mce", "brier_reliability", "brier_resolution", "accuracy"):
        assert a[k] == pytest.approx(b[k]), k
    assert set(evaluate(fl)) == {"n_fights", "accuracy", "probabilistic", "discrimination", "calibration"}


# --------------------------------------------------------------------------- fold isolation

def test_scoring_refuses_rows_the_model_or_imputer_could_have_seen():
    rows = make_rows(400)
    cut = rows.event_date.sort_values().iloc[len(rows) // 2]
    train, later = rows[rows.event_date < cut], rows[rows.event_date >= cut]
    m = LogisticModel([c for c in NUMERIC if c.startswith("diff_")], 1.0, 0).fit(train)
    assert m.physical_.fitted_until_ == train.event_date.max() and m.physical_.n_fit_rows_ == len(train)
    score(m, later)                                     # strictly later: fine
    with pytest.raises(FoldLeakError):
        score(m, rows)                                  # overlaps the training window
    base = BaselineModel().fit(train)
    with pytest.raises(FoldLeakError):
        score(base, train)


def test_imputer_ignores_evaluation_rows():
    rows = make_rows(400)
    cut = rows.event_date.sort_values().iloc[len(rows) // 2]
    train, later = rows[rows.event_date < cut].copy(), rows[rows.event_date >= cut].copy()
    m = LogisticModel([c for c in NUMERIC if c.startswith("diff_")], 1.0, 0).fit(train)
    before = (m.physical_.reach_slope_, dict(m.physical_.median_))
    later["f1_reach_in"] = 999.0                        # absurd eval values must not move the fit
    score(m, later)
    assert (m.physical_.reach_slope_, dict(m.physical_.median_)) == before


# --------------------------------------------------------------------------- walk-forward backtest

SMALL_MCFG = {"random_seed": 0, "logistic": {"C_grid": [0.1, 1.0]},
              "lightgbm": {"learning_rate": 0.1, "max_estimators": 50, "early_stopping_rounds": 10,
                           "grid": {"num_leaves": [7], "min_child_samples": [20]}, "fixed": {}},
              "calibration_bins": [0.5, 0.6, 0.7, 1.0]}


def test_backtest_runs_rolling_origin_slices():
    rows = make_rows(1500)   # daily dates 2010-01-01 .. 2014-02
    fl = {"numeric": NUMERIC, "categorical": CATEGORICAL, "features_sha256": "abc", "data_cutoff": "2014-02-08"}
    bcfg = {"inner_validation_years": 1, "slices": [{"train_end": "2012-01-01", "test_end": "2013-01-01"},
                                                     {"train_end": "2013-01-01", "test_end": None}]}
    rep = run_backtest(rows, fl, SMALL_MCFG, bcfg, log=False)
    assert rep["features_sha256"] == "abc" and len(rep["slices"]) == 2
    s1, s2 = rep["slices"]
    assert s1["cutoff"] == "2012-01-01" and s1["test_period"] == "2012-01-01..2012-12-31"
    assert s2["test_period"].endswith(str(rows.event_date.max().date()))
    for s in rep["slices"]:
        assert all(pd.Timestamp(d) < pd.Timestamp(s["cutoff"]) for d in s["imputer_fitted_until"].values())
        assert s["metrics"]["lightgbm"]["discrimination"]["roc_auc"] > 0.7
        assert set(s["calibration"]) == {"baseline", "logistic", "lightgbm"}
    summary = backtest_summary(rep)
    assert list(summary.columns[:8]) == ["Cutoff", "Test Period", "Model", "N", "AUC", "Brier", "LogLoss", "ECE"]
    assert len(summary) == 6


# --------------------------------------------------------------------------- promotion gate

def test_promotion_gates(tmp_path):
    from src.features import file_sha256
    (tmp_path / "fight_features.csv").write_text("a,b\n1,2\n")
    sha = file_sha256(tmp_path / "fight_features.csv")
    fl = {"provenance_checked": True, "features_sha256": sha}
    bt = {"features_sha256": sha, "slices": [{}]}
    ok = {"passed": True, "summary": "101 passed"}
    assert promotion_gates(fl, tmp_path, bt, ok) == []
    assert promotion_gates({**fl, "provenance_checked": False}, tmp_path, bt, ok)[0].startswith("provenance")
    assert promotion_gates(fl, tmp_path, {"features_sha256": "other", "slices": [{}]}, ok)[0].startswith("backtest")
    assert promotion_gates(fl, tmp_path, None, ok) == ["backtest: not run"]
    assert promotion_gates(fl, tmp_path, bt, {"passed": False, "summary": "1 failed"})[0].startswith("tests")
    assert promotion_gates(fl, tmp_path, bt, None) == ["tests: not run"]
    (tmp_path / "fight_features.csv").write_text("a,b\n1,3\n")   # changed after the provenance check
    assert any(f.startswith("provenance") for f in promotion_gates(fl, tmp_path, bt, ok))


def test_lgbm_unseen_category_becomes_nan():
    train, test = make_rows(300, 0), make_rows(20, 1)
    test.loc[0, "weight_class"] = "Brand New Division"
    m = LGBMModel(NUMERIC, CATEGORICAL, {"num_leaves": 7, "random_state": 0}, 30).fit(train)
    X = m._frame(test)
    assert pd.isna(X.loc[0, "weight_class"]) and len(m.predict_proba(test)) == len(test)
