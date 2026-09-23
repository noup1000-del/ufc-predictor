"""Stage 4: train and evaluate models on a strictly time-based split.

Models (fit on `train`, tuned on `valid`, reported on `valid` and `test`):
  1. baseline  - pick the fighter with more UFC wins (coin flip on ties)
  2. logistic  - median imputation (fit on train) + standardisation + logistic regression on diff_ features
  3. lightgbm  - all numeric + categorical features, native NaN handling, light grid on validation

All metrics are per fight, using the symmetric probability
p = (p(A,B) + 1 - p(B,A)) / 2 over both orientations, and are grouped as
overall probabilistic (Brier, log loss), discrimination (ROC-AUC, Brier resolution) and
calibration (ECE, MCE, Brier reliability, bin table with 95% Wilson intervals).

Fold isolation: every model (and its PhysicalImputer) records the last event date it was
fitted on; `score()` refuses to evaluate rows that are not strictly later.

The model with the best validation log loss (logistic or lightgbm) is refit on all data
and saved to models/model_<YYYY-MM-DD>.pkl (+ .json). It is promoted to models/latest.pkl
only if the gates pass: features provenance-checked and unchanged, walk-forward backtest
run on the same features (models/backtest_report.json), and the full test suite green.

Usage: python -m src.train [--backtest]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import pickle
import shutil
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config, resolve_path
from src.features import symmetric_probability

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- data

def load_features(features_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(features_dir / "fight_features.csv", parse_dates=["event_date"],
                     dtype={"fight_id": str, "fighter_1_id": str, "fighter_2_id": str})
    feature_list = json.loads((features_dir / "feature_list.json").read_text(encoding="utf-8"))
    return df, feature_list


def fight_level(rows: pd.DataFrame, p_rows: np.ndarray) -> pd.DataFrame:
    """Combine both orientations of each fight into one symmetric probability for fighter_1
    of orientation 0."""
    r = rows[["fight_id", "event_date", "orientation", "target"]].copy()
    r["p"] = p_rows
    o0 = r[r.orientation == 0].set_index("fight_id")
    o1 = r[r.orientation == 1].set_index("fight_id")
    if set(o0.index) != set(o1.index) or o0.index.duplicated().any():
        raise ValueError("every fight needs exactly one row per orientation")
    o1 = o1.loc[o0.index]
    return pd.DataFrame({"fight_id": o0.index, "event_date": o0["event_date"].to_numpy(),
                         "target": o0["target"].to_numpy(),
                         "p": symmetric_probability(o0["p"].to_numpy(), o1["p"].to_numpy())})


def _coin(fight_ids) -> np.ndarray:
    """Deterministic coin flip per fight (for exact 0.5 ties)."""
    return np.array([zlib.crc32(str(f).encode()) % 2 for f in fight_ids])


DEFAULT_BINS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 1.0]
_EPS = 1e-6


def wilson_interval(k, n, z: float = 1.96):
    """95% Wilson score interval for k successes out of n (vectorised; NaN where n == 0)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        phat = k / n
        denom = 1 + z ** 2 / n
        centre = (phat + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    lo, hi = centre - half, centre + half
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def _folded(fl: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Favourite's probability (>= 0.5) and whether the favourite won. Folding makes the
    calibration views independent of the arbitrary fighter order; Brier is unchanged by it."""
    p = np.clip(fl["p"].to_numpy(dtype=float), _EPS, 1 - _EPS)
    y = fl["target"].to_numpy(dtype=float)
    return np.maximum(p, 1 - p), np.where(p >= 0.5, y, 1 - y)


def calibration_table(fl: pd.DataFrame, bins: list[float] = DEFAULT_BINS) -> pd.DataFrame:
    """Per bin of the favourite's predicted probability: n, mean prediction, actual win rate
    with a 95% Wilson interval, and the gap."""
    q, o = _folded(fl)
    b = pd.cut(q, bins=bins, include_lowest=True, right=False)
    t = pd.DataFrame({"bin": b, "p": q, "won": o}).groupby("bin", observed=False).agg(
        n=("p", "size"), mean_predicted=("p", "mean"), wins=("won", "sum"), actual_win_rate=("won", "mean"))
    t["lower_ci"], t["upper_ci"] = wilson_interval(t["wins"], t["n"])
    t["gap"] = t["actual_win_rate"] - t["mean_predicted"]
    return t.drop(columns="wins").reset_index()


def calibration_errors(table: pd.DataFrame) -> tuple[float, float]:
    """ECE = sum_k n_k/N |gap_k|;  MCE = max_k |gap_k| over non-empty bins."""
    t = table[table["n"] > 0]
    gaps = (t["actual_win_rate"] - t["mean_predicted"]).abs()
    return float((t["n"] * gaps).sum() / t["n"].sum()), float(gaps.max())


def brier_decomposition(fl: pd.DataFrame, bins: list[float] = DEFAULT_BINS) -> dict:
    """Murphy decomposition on the binned, favourite-folded forecasts:
    Brier ~= reliability - resolution + uncertainty (exact when forecasts are constant within bins)."""
    q, o = _folded(fl)
    b = pd.cut(q, bins=bins, include_lowest=True, right=False)
    g = pd.DataFrame({"b": b, "q": q, "o": o}).groupby("b", observed=True).agg(
        n=("q", "size"), q=("q", "mean"), o=("o", "mean"))
    n, obar = g["n"].sum(), o.mean()
    return {"reliability": float((g["n"] * (g["q"] - g["o"]) ** 2).sum() / n),
            "resolution": float((g["n"] * (g["o"] - obar) ** 2).sum() / n),
            "uncertainty": float(obar * (1 - obar))}


def evaluate(fl: pd.DataFrame, bins: list[float] = DEFAULT_BINS) -> dict:
    """Metric taxonomy for fight-level symmetric probabilities."""
    y = fl["target"].to_numpy(dtype=float)
    p = np.clip(fl["p"].to_numpy(dtype=float), _EPS, 1 - _EPS)
    pred = np.where(p > 0.5, 1, np.where(p < 0.5, 0, _coin(fl["fight_id"])))
    # AUC over both orientations: invariant to which fighter is listed first.
    y2, p2 = np.r_[y, 1 - y], np.r_[p, 1 - p]
    auc = float(roc_auc_score(y2, p2)) if len(np.unique(y2)) == 2 else float("nan")
    decomp = brier_decomposition(fl, bins)
    ece, mce = calibration_errors(calibration_table(fl, bins))
    return {
        "n_fights": int(len(fl)),
        "accuracy": float((pred == y).mean()),
        "probabilistic": {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
                          "brier_uncertainty": decomp["uncertainty"]},
        "discrimination": {"roc_auc": auc, "brier_resolution": decomp["resolution"]},
        "calibration": {"ece": ece, "mce": mce, "brier_reliability": decomp["reliability"]},
    }


def flat_metrics(m: dict) -> dict:
    """{'n_fights', 'accuracy', 'brier', 'log_loss', 'roc_auc', ...} for tables."""
    out = {"n_fights": m["n_fights"], "accuracy": m["accuracy"]}
    for group in ("probabilistic", "discrimination", "calibration"):
        out.update(m[group])
    return out


# --------------------------------------------------------------------------- physical-data imputation

PHYSICAL = ["height_in", "reach_in", "age"]


class PhysicalImputer:
    """Fill missing height/reach/age with values fitted on the training rows.

    Why: ufcstats bios are filled in over time for fighters who stay in the UFC, so in
    historical data a *missing* reach/height/DOB mostly marks short careers (losers).
    Left as NaN, that missingness leaks the fighter's future. Imputing removes the signal.
    reach <- linear fit on height, else weight-class median; height <- weight-class
    median; age <- median. `diff_` columns are recomputed after imputation.
    """

    def fit(self, rows: pd.DataFrame) -> "PhysicalImputer":
        # Fit window, used to prove it never saw evaluation rows (see assert_fit_before).
        self.fitted_until_ = pd.Timestamp(rows["event_date"].max()) if "event_date" in rows else None
        self.n_fit_rows_ = int(len(rows))
        long = pd.concat([pd.DataFrame({"wc": rows["weight_class"].astype("string").fillna("?"),
                                        **{p: rows[f"f{s}_{p}"] for p in PHYSICAL}}) for s in ("1", "2")],
                         ignore_index=True)
        both = long.dropna(subset=["height_in", "reach_in"])
        self.reach_slope_, self.reach_intercept_ = np.polyfit(both["height_in"], both["reach_in"], 1)
        self.wc_median_ = long.groupby("wc")[PHYSICAL].median().to_dict()
        self.median_ = long[PHYSICAL].median().to_dict()
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy()
        wc = rows["weight_class"].astype("string").fillna("?")
        for s in ("1", "2"):
            h, r, a = (out[f"f{s}_{p}"].astype(float) for p in PHYSICAL)
            h = h.fillna(wc.map(self.wc_median_["height_in"]).astype(float)).fillna(self.median_["height_in"])
            r = r.fillna(self.reach_intercept_ + self.reach_slope_ * out[f"f{s}_height_in"].astype(float))
            r = r.fillna(wc.map(self.wc_median_["reach_in"]).astype(float)).fillna(self.median_["reach_in"])
            a = a.fillna(wc.map(self.wc_median_["age"]).astype(float)).fillna(self.median_["age"])
            out[f"f{s}_height_in"], out[f"f{s}_reach_in"], out[f"f{s}_age"] = h, r, a
        for p in PHYSICAL:
            out[f"diff_{p}"] = out[f"f1_{p}"] - out[f"f2_{p}"]
        return out


class FoldLeakError(ValueError):
    """A model (or its imputer) was fitted on rows that are not strictly before the eval rows."""


def fitted_until(model) -> pd.Timestamp | None:
    imp = getattr(model, "physical_", None)
    if imp is not None:
        return getattr(imp, "fitted_until_", None)
    return getattr(model, "fitted_until_", None)


def assert_fit_before(model, rows: pd.DataFrame) -> None:
    """Fold-strict isolation: model + imputer must be fitted only on data before the eval window."""
    until = fitted_until(model)
    if until is None:
        raise FoldLeakError(f"{model.name}: unknown fit window; refusing to evaluate")
    first = pd.Timestamp(rows["event_date"].min())
    if until >= first:
        raise FoldLeakError(f"{model.name}: fitted on data up to {until.date()}, "
                            f"but evaluation starts {first.date()}")


def score(model, rows: pd.DataFrame, bins: list[float] = DEFAULT_BINS) -> tuple[dict, pd.DataFrame]:
    """The only way evaluation metrics are computed: checks fold isolation, then scores."""
    assert_fit_before(model, rows)
    fl = fight_level(rows, model.predict_proba(rows))
    return evaluate(fl, bins), calibration_table(fl, bins)


def missingness_report(rows: pd.DataFrame, columns: list[str], min_share=0.01) -> pd.DataFrame:
    """Target rate when a fighter_1 feature is missing vs present (orientation-balanced rows).
    A rate far from 0.5 for data that should be known pre-fight hints at leakage."""
    out = []
    for c in columns:
        miss = rows[c].isna()
        if miss.mean() >= min_share and miss.sum() >= 30:
            out.append({"feature": c, "missing_share": miss.mean(),
                        "target_rate_missing": rows.loc[miss, "target"].mean(),
                        "target_rate_present": rows.loc[~miss, "target"].mean()})
    return pd.DataFrame(out, columns=["feature", "missing_share", "target_rate_missing", "target_rate_present"])


# --------------------------------------------------------------------------- models

class BaselineModel:
    """More UFC wins -> favourite. Probability = the rule's accuracy on train (0.5 on ties)."""
    name = "baseline"

    def fit(self, rows: pd.DataFrame) -> "BaselineModel":
        self.fitted_until_ = pd.Timestamp(rows["event_date"].max())
        o0 = rows[rows.orientation == 0]
        d = o0["f1_n_wins"] - o0["f2_n_wins"]
        decided = d != 0
        self.p_favourite_ = float(((d[decided] > 0).astype(int) == o0.loc[decided, "target"]).mean())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        d = (rows["f1_n_wins"] - rows["f2_n_wins"]).to_numpy()
        return np.where(d > 0, self.p_favourite_, np.where(d < 0, 1 - self.p_favourite_, 0.5))

    def params(self) -> dict:
        return {"p_favourite": self.p_favourite_}


class LogisticModel:
    name = "logistic"

    def __init__(self, columns: list[str], C: float, seed: int):
        self.columns, self.C, self.seed = columns, C, seed

    def fit(self, rows: pd.DataFrame) -> "LogisticModel":
        self.physical_ = PhysicalImputer().fit(rows)
        X = self.physical_.transform(rows)[self.columns].to_numpy(dtype=float)
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),  # medians from training rows only
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed)),
        ]).fit(X, rows["target"].to_numpy())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        X = self.physical_.transform(rows)[self.columns].to_numpy(dtype=float)
        return self.pipe_.predict_proba(X)[:, 1]

    def params(self) -> dict:
        return {"C": self.C}

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-feature log-odds contributions (coef x standardised value) plus `_bias`.
        Row sums equal the model's log-odds."""
        X = self.physical_.transform(rows)[self.columns].to_numpy(dtype=float)
        z = self.pipe_[:-1].transform(X)
        lr = self.pipe_.named_steps["lr"]
        out = pd.DataFrame(z * lr.coef_[0], columns=self.columns, index=rows.index)
        out["_bias"] = lr.intercept_[0]
        return out

    def coefficients(self) -> pd.Series:
        return pd.Series(self.pipe_.named_steps["lr"].coef_[0], index=self.columns)


class LGBMModel:
    name = "lightgbm"

    def __init__(self, numeric: list[str], categorical: list[str], params: dict, n_estimators: int):
        self.numeric, self.categorical = numeric, categorical
        self.params_in, self.n_estimators = params, n_estimators

    def _frame(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = self.physical_.transform(rows)
        X = rows[self.numeric].astype(float).copy()
        for c in self.categorical:
            cats = self.categories_[c]
            v = rows[c].astype("string")
            X[c] = pd.Categorical(v.where(v.isin(cats)), categories=cats)  # unseen level -> NaN
        return X

    def fit(self, rows: pd.DataFrame, valid: pd.DataFrame | None = None,
            early_stopping_rounds: int | None = None) -> "LGBMModel":
        # Imputation values and category levels come from the training rows only.
        self.physical_ = PhysicalImputer().fit(rows)
        self.categories_ = {c: sorted(rows[c].dropna().astype(str).unique()) for c in self.categorical}
        self.model_ = lgb.LGBMClassifier(n_estimators=self.n_estimators, verbose=-1, **self.params_in)
        kwargs = {}
        if valid is not None:
            kwargs = {"eval_X": (self._frame(valid),), "eval_y": (valid["target"],),
                      "eval_metric": "binary_logloss",
                      "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)]}
        self.model_.fit(self._frame(rows), rows["target"], **kwargs)
        self.best_iteration_ = int(self.model_.best_iteration_ or self.n_estimators)
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._frame(rows), num_iteration=self.best_iteration_)[:, 1]

    def params(self) -> dict:
        return {**self.params_in, "n_estimators": self.best_iteration_}

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """LightGBM SHAP-style contributions (log-odds) per feature plus `_bias`.
        Row sums equal the model's raw score (log-odds)."""
        X = self._frame(rows)
        c = self.model_.booster_.predict(X, pred_contrib=True, num_iteration=self.best_iteration_)
        return pd.DataFrame(c, columns=list(X.columns) + ["_bias"], index=rows.index)

    def importances(self) -> pd.Series:
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        s = pd.Series(gain, index=self.model_.booster_.feature_name()).sort_values(ascending=False)
        return s / s.sum()


# --------------------------------------------------------------------------- tuning

def _valid_log_loss(model, valid: pd.DataFrame) -> float:
    return score(model, valid)[0]["probabilistic"]["log_loss"]


def tune_logistic(train, valid, columns, grid, seed, log: bool = True):
    results = []
    for C in grid:
        m = LogisticModel(columns, C, seed).fit(train)
        results.append((_valid_log_loss(m, valid), C, m))
    if log:
        for ll, C, _ in sorted(results, key=lambda r: r[1]):
            logger.info("  logistic C=%-5g valid log loss %.4f", C, ll)
    return min(results, key=lambda r: r[0])[2]


def tune_lightgbm(train, valid, numeric, categorical, cfg, seed, log: bool = True):
    lcfg = cfg["lightgbm"]
    grid = lcfg["grid"]
    best = None
    for values in itertools.product(*grid.values()):
        params = {**lcfg["fixed"], **dict(zip(grid.keys(), values)),
                  "learning_rate": lcfg["learning_rate"], "random_state": seed}
        m = LGBMModel(numeric, categorical, params, lcfg["max_estimators"]).fit(
            train, valid, lcfg["early_stopping_rounds"])
        ll = _valid_log_loss(m, valid)
        if log:
            logger.info("  lightgbm %s trees=%d valid log loss %.4f",
                        dict(zip(grid.keys(), values)), m.best_iteration_, ll)
        if best is None or ll < best[0]:
            best = (ll, m)
    return best[1]


def fit_candidates(train, valid, numeric, categorical, mcfg, log: bool = True) -> dict:
    """Baseline, logistic and LightGBM fitted on `train`, hyperparameters chosen on `valid`."""
    seed = mcfg["random_seed"]
    diff_cols = [c for c in numeric if c.startswith("diff_")]
    return {"baseline": BaselineModel().fit(train),
            "logistic": tune_logistic(train, valid, diff_cols, mcfg["logistic"]["C_grid"], seed, log),
            "lightgbm": tune_lightgbm(train, valid, numeric, categorical, mcfg, seed, log)}


def refit(model, rows: pd.DataFrame):
    """Same model configuration (tuned hyperparameters, tree count) refit on `rows`."""
    if isinstance(model, BaselineModel):
        return BaselineModel().fit(rows)
    if isinstance(model, LogisticModel):
        return LogisticModel(model.columns, model.C, model.seed).fit(rows)
    return LGBMModel(model.numeric, model.categorical, model.params_in, model.best_iteration_).fit(rows)


def flag_dominance(importances: pd.Series, threshold: float) -> list[str]:
    return [f"{name} has {share:.0%} of total gain (> {threshold:.0%}): check for leakage"
            for name, share in importances.items() if share > threshold]


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")


# --------------------------------------------------------------------------- walk-forward backtest

def run_backtest(df: pd.DataFrame, feature_list: dict, mcfg: dict, bcfg: dict, log: bool = True) -> dict:
    """Rolling-origin backtest. Per slice: tune on the last `inner_validation_years` before the
    cutoff, refit on everything before the cutoff (imputer included), score the test window."""
    numeric, categorical = feature_list["numeric"], feature_list["categorical"]
    bins = mcfg.get("calibration_bins", DEFAULT_BINS)
    slices = []
    for sl in bcfg["slices"]:
        cutoff = pd.Timestamp(sl["train_end"])
        test_end = pd.Timestamp(sl["test_end"]) if sl.get("test_end") else None
        train_all = df[df.event_date < cutoff].reset_index(drop=True)
        test = df[(df.event_date >= cutoff) & ((df.event_date < test_end) if test_end is not None else True)]
        test = test.reset_index(drop=True)
        inner_cut = cutoff - pd.DateOffset(years=bcfg.get("inner_validation_years", 1))
        inner_train = train_all[train_all.event_date < inner_cut]
        inner_valid = train_all[train_all.event_date >= inner_cut]
        if log:
            logger.info("Backtest slice: train < %s (tune on %s..%s), test %s..%s (%d fights)",
                        cutoff.date(), inner_cut.date(), (cutoff - pd.Timedelta(days=1)).date(),
                        test.event_date.min().date(), test.event_date.max().date(), test.fight_id.nunique())
        tuned = fit_candidates(inner_train, inner_valid, numeric, categorical, mcfg, log=False)
        models = {name: refit(m, train_all) for name, m in tuned.items()}
        metrics, calib = {}, {}
        for name, m in models.items():
            metrics[name], table = score(m, test, bins)
            calib[name] = table.astype({"bin": str}).to_dict("records")
        slices.append({
            "cutoff": str(cutoff.date()),
            "test_period": f"{test.event_date.min().date()}..{test.event_date.max().date()}",
            "n_train_fights": int(train_all.fight_id.nunique()),
            "n_test_fights": int(test.fight_id.nunique()),
            "imputer_fitted_until": {n: str(fitted_until(m).date()) for n, m in models.items()},
            "params": {n: m.params() for n, m in models.items()},
            "metrics": metrics, "calibration": calib,
        })
    return {"generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "features_sha256": feature_list.get("features_sha256"), "data_cutoff": feature_list.get("data_cutoff"),
            "inner_validation_years": bcfg.get("inner_validation_years", 1), "slices": slices}


def backtest_summary(report: dict) -> pd.DataFrame:
    rows = []
    for s in report["slices"]:
        for model, m in s["metrics"].items():
            rows.append({"Cutoff": s["cutoff"], "Test Period": s["test_period"], "Model": model,
                         "N": m["n_fights"], "AUC": m["discrimination"]["roc_auc"],
                         "Brier": m["probabilistic"]["brier"], "LogLoss": m["probabilistic"]["log_loss"],
                         "ECE": m["calibration"]["ece"], "Acc": m["accuracy"]})
    return pd.DataFrame(rows)


def log_backtest(report: dict, models: tuple[str, ...] = ("logistic", "lightgbm")) -> None:
    logger.info("Walk-forward backtest (per fight, symmetric probabilities):\n%s",
                _fmt_table(backtest_summary(report)))
    for s in report["slices"]:
        for name in models:
            t = pd.DataFrame(s["calibration"][name])[
                ["bin", "n", "mean_predicted", "actual_win_rate", "lower_ci", "upper_ci", "gap"]]
            logger.info("Calibration, %s, cutoff %s (test %s):\n%s", name, s["cutoff"], s["test_period"],
                        _fmt_table(t))


def run_backtest_cli(save: bool = True) -> dict:
    cfg = load_config()
    features_dir = resolve_path(cfg["paths"]["features_dir"])
    df, fl = load_features(features_dir)
    _check_features_provenance(fl, features_dir)
    report = run_backtest(df, fl, cfg["model"], cfg["backtest"])
    log_backtest(report)
    if save:
        path = resolve_path(cfg["paths"]["models_dir"]) / "backtest_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %s", path)
    return report


# --------------------------------------------------------------------------- promotion gate

class PromotionError(RuntimeError):
    """The candidate model did not pass the promotion gates; latest.pkl was not changed."""


def _check_features_provenance(fl: dict, features_dir: Path) -> None:
    from src.features import file_sha256
    if not fl.get("provenance_checked"):
        raise PromotionError("features were not built with the provenance check; rerun the features stage")
    actual = file_sha256(features_dir / "fight_features.csv")
    if fl.get("features_sha256") != actual:
        raise PromotionError("fight_features.csv changed after its provenance check; rerun the features stage")


def run_test_suite(timeout: int) -> dict:
    """Run the full pytest suite in a subprocess (promotion gate)."""
    import subprocess
    from src.config import PROJECT_ROOT
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, timeout=timeout)
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    return {"passed": proc.returncode == 0, "returncode": proc.returncode, "summary": last}


def promotion_gates(fl: dict, features_dir: Path, backtest: dict, tests: dict | None) -> list[str]:
    """Return the list of failed gates (empty = promote)."""
    failed = []
    try:
        _check_features_provenance(fl, features_dir)
    except PromotionError as e:
        failed.append(f"provenance: {e}")
    if not backtest or len(backtest.get("slices", [])) == 0:
        failed.append("backtest: not run")
    elif backtest.get("features_sha256") != fl.get("features_sha256"):
        failed.append("backtest: run on different features than the candidate")
    if tests is None:
        failed.append("tests: not run")
    elif not tests["passed"]:
        failed.append(f"tests: failed ({tests['summary']})")
    return failed


# --------------------------------------------------------------------------- training

def run_train(today: str | None = None, run_tests: bool | None = None) -> dict:
    cfg = load_config()
    mcfg = cfg["model"]
    bins = mcfg.get("calibration_bins", DEFAULT_BINS)
    today = today or datetime.now().strftime("%Y-%m-%d")
    features_dir = resolve_path(cfg["paths"]["features_dir"])
    df, fl = load_features(features_dir)
    _check_features_provenance(fl, features_dir)

    numeric, categorical = fl["numeric"], fl["categorical"]
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    for s, d in splits.items():
        o0 = d[d.orientation == 0]
        logger.info("%-5s %5d fights  %s -> %s", s, len(o0), o0.event_date.min().date(), o0.event_date.max().date())
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    if not (train.event_date.max() < valid.event_date.min() and valid.event_date.max() < test.event_date.min()):
        raise FoldLeakError("train/valid/test splits are not in time order")

    miss = missingness_report(train, [c for c in numeric if c.startswith("f1_")])
    if len(miss):
        logger.info("Missingness vs target on train (fighter_1 features):\n%s", _fmt_table(miss))
    phys_cols = {f"f1_{p}" for p in PHYSICAL}
    for r in miss[(miss.target_rate_missing - 0.5).abs() > 0.25].itertuples():
        handled = " (handled: imputed before modelling)" if r.feature in phys_cols else ""
        logger.warning("Missing %s coincides with target rate %.2f%s", r.feature, r.target_rate_missing, handled)

    logger.info("Tuning on validation (imputer and models fitted on train only)")
    cands = fit_candidates(train, valid, numeric, categorical, mcfg)
    logistic, lgbm = cands["logistic"], cands["lightgbm"]

    results, calib = {}, {}
    for name, m in cands.items():
        results[name] = {}
        for s in ("valid", "test"):
            results[name][s], calib[(name, s)] = score(m, splits[s], bins)

    comparison = pd.DataFrame([{"model": n, "split": s, **flat_metrics(r[s])}
                               for n, r in results.items() for s in ("valid", "test")])
    groups = {"overall probabilistic": ["brier", "log_loss", "accuracy"],
              "discrimination": ["roc_auc", "brier_resolution"],
              "calibration": ["ece", "mce", "brier_reliability"]}
    for title, cols in groups.items():
        logger.info("Model comparison: %s\n%s", title, _fmt_table(comparison[["model", "split", "n_fights"] + cols]))
    for (name, s), t in calib.items():
        if s == "test":
            logger.info("Calibration, %s on test (favourite's p vs actual win rate, 95%% Wilson CI):\n%s",
                        name, _fmt_table(t))

    imp = lgbm.importances()
    logger.info("LightGBM gain importance (top 20):\n%s",
                imp.head(20).to_frame("gain_share").to_string(float_format=lambda x: f"{x:.3f}"))
    coefs = logistic.coefficients().sort_values(key=abs, ascending=False)
    logger.info("Logistic standardised coefficients (top 15):\n%s",
                coefs.head(15).to_frame("coef").to_string(float_format=lambda x: f"{x:+.3f}"))
    flags = flag_dominance(imp, mcfg["dominance_threshold"])
    for f in flags:
        logger.warning("Possible leakage: %s", f)
    if not flags:
        logger.info("No feature above the %.0f%% dominance threshold", 100 * mcfg["dominance_threshold"])

    # Production candidate: best validation log loss, refit on all data.
    chosen = min((logistic, lgbm), key=lambda m: results[m.name]["valid"]["probabilistic"]["log_loss"])
    all_rows = pd.concat([train, valid, test], ignore_index=True)
    prod = refit(chosen, all_rows)
    logger.info("Candidate model: %s %s, refit on %d fights up to %s", chosen.name, prod.params(),
                all_rows.fight_id.nunique(), all_rows.event_date.max().date())

    # Gate 1+2: walk-forward backtest on the same features (always run before promotion).
    backtest = run_backtest(df, fl, mcfg, cfg["backtest"])
    log_backtest(backtest)
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    bt_text = json.dumps(backtest, indent=2, default=str)
    (models_dir / "backtest_report.json").write_text(bt_text, encoding="utf-8")
    (models_dir / f"backtest_{today}.json").write_text(bt_text, encoding="utf-8")

    # Gate 3: the full test suite (includes the leakage/provenance tests).
    run_tests = mcfg.get("promotion", {}).get("run_tests", True) if run_tests is None else run_tests
    tests = run_test_suite(mcfg.get("promotion", {}).get("tests_timeout_sec", 900)) if run_tests else None
    if tests:
        logger.info("Test suite: %s", tests["summary"])
    failed = promotion_gates(fl, features_dir, backtest, tests)

    manifest_path = resolve_path(cfg["source"]["source_dir"]) / "manifest.json"
    source = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    metadata = {
        "model_version": f"model_{today}",
        "model_type": chosen.name,
        "params": prod.params(),
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_cutoff": str(all_rows.event_date.max().date()),
        "n_fights_trained": int(all_rows.fight_id.nunique()),
        "source_commit": source.get("commit_sha"),
        "features_sha256": fl.get("features_sha256"),
        "split": cfg["split"],
        "evaluation": {"note": "models (incl. imputer) fit on train only, tuned on valid; "
                               "production model refit on all data",
                       "metrics": results,
                       "calibration_test": {n: calib[(n, "test")].astype({"bin": str}).to_dict("records")
                                            for n in results}},
        "backtest": {"report": f"backtest_{today}.json",
                     "summary": backtest_summary(backtest).round(4).to_dict("records")},
        "selection": "lowest validation log loss among logistic and lightgbm",
        "features": {"numeric": numeric if chosen is lgbm else logistic.columns,
                     "categorical": categorical if chosen is lgbm else []},
        "importance_flags": flags,
        "lightgbm_top_importance": imp.head(20).round(4).to_dict(),
        "promotion": {"tests": tests, "failed_gates": failed, "promoted": not failed},
    }
    pkl = models_dir / f"model_{today}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"model": prod, "metadata": metadata}, f)
    (models_dir / f"model_{today}.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if failed:
        raise PromotionError(f"{pkl.name} saved but NOT promoted to latest.pkl: {failed}")
    # Copies rather than symlinks (symlinks need admin rights on Windows).
    shutil.copyfile(pkl, models_dir / "latest.pkl")
    shutil.copyfile(models_dir / f"model_{today}.json", models_dir / "latest.json")
    logger.info("Promoted %s to models/latest.pkl (all gates passed)", pkl.name)
    return metadata


def load_model(path: Path | None = None) -> dict:
    path = path or resolve_path(load_config()["paths"]["models_dir"]) / "latest.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backtest", action="store_true", help="only run the walk-forward backtest")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_backtest_cli() if args.backtest else run_train()
    except Exception:
        logger.exception("Training failed")
        return 1
    return 0


if __name__ == "__main__":
    # Run via the importable module so pickled model classes are stored as `src.train.*`,
    # not `__main__.*` (which could not be unpickled by predict.py).
    from src import train as _train_module
    sys.exit(_train_module.main())
