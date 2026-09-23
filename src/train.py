"""Stage 4: train and evaluate models on a strictly time-based split.

Models (fit on `train`, tuned on `valid`, reported on `valid` and `test`):
  1. baseline  - pick the fighter with more UFC wins (coin flip on ties)
  2. logistic  - median imputation (fit on train) + standardisation + logistic regression on diff_ features
  3. lightgbm  - all numeric + categorical features, native NaN handling, light grid on validation

All metrics are per fight, using the symmetric probability
p = (p(A,B) + 1 - p(B,A)) / 2 over both orientations.

The model with the best validation log loss (logistic or lightgbm) is refit on all
data and saved to models/model_<YYYY-MM-DD>.pkl (+ .json metadata) and models/latest.pkl.

Usage: python -m src.train
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
from sklearn.metrics import brier_score_loss, log_loss
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


def evaluate(fl: pd.DataFrame) -> dict:
    y, p = fl["target"].to_numpy(), np.clip(fl["p"].to_numpy(), 1e-6, 1 - 1e-6)
    pred = np.where(p > 0.5, 1, np.where(p < 0.5, 0, _coin(fl["fight_id"])))
    return {"n_fights": int(len(fl)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "accuracy": float((pred == y).mean())}


def calibration_table(fl: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    """Folded to the favourite: predicted probability of the favourite vs how often it won."""
    p, y = fl["p"].to_numpy(), fl["target"].to_numpy()
    fav_p = np.maximum(p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    b = pd.cut(fav_p, bins=bins, include_lowest=True, right=False)
    t = pd.DataFrame({"bin": b, "p": fav_p, "won": fav_won}).groupby("bin", observed=False).agg(
        n=("p", "size"), mean_predicted=("p", "mean"), actual_win_rate=("won", "mean"))
    t["gap"] = t["actual_win_rate"] - t["mean_predicted"]
    return t.reset_index()


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

    def importances(self) -> pd.Series:
        gain = self.model_.booster_.feature_importance(importance_type="gain")
        s = pd.Series(gain, index=self.model_.booster_.feature_name()).sort_values(ascending=False)
        return s / s.sum()


# --------------------------------------------------------------------------- training

def tune_logistic(train, valid, columns, grid, seed):
    results = []
    for C in grid:
        m = LogisticModel(columns, C, seed).fit(train)
        results.append((evaluate(fight_level(valid, m.predict_proba(valid)))["log_loss"], C, m))
    results.sort(key=lambda r: r[0])
    for ll, C, _ in sorted(results, key=lambda r: r[1]):
        logger.info("  logistic C=%-5g valid log loss %.4f", C, ll)
    return results[0][2]


def tune_lightgbm(train, valid, numeric, categorical, cfg, seed):
    lcfg = cfg["lightgbm"]
    grid = lcfg["grid"]
    best = None
    for values in itertools.product(*grid.values()):
        params = {**lcfg["fixed"], **dict(zip(grid.keys(), values)),
                  "learning_rate": lcfg["learning_rate"], "random_state": seed}
        m = LGBMModel(numeric, categorical, params, lcfg["max_estimators"]).fit(
            train, valid, lcfg["early_stopping_rounds"])
        ll = evaluate(fight_level(valid, m.predict_proba(valid)))["log_loss"]
        logger.info("  lightgbm %s trees=%d valid log loss %.4f",
                    dict(zip(grid.keys(), values)), m.best_iteration_, ll)
        if best is None or ll < best[0]:
            best = (ll, m)
    return best[1]


def flag_dominance(importances: pd.Series, threshold: float) -> list[str]:
    return [f"{name} has {share:.0%} of total gain (> {threshold:.0%}): check for leakage"
            for name, share in importances.items() if share > threshold]


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def run_train(today: str | None = None) -> dict:
    cfg = load_config()
    mcfg = cfg["model"]
    seed = mcfg["random_seed"]
    today = today or datetime.now().strftime("%Y-%m-%d")
    df, fl = load_features(resolve_path(cfg["paths"]["features_dir"]))

    numeric, categorical = fl["numeric"], fl["categorical"]
    diff_cols = [c for c in numeric if c.startswith("diff_")]
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    for s, d in splits.items():
        o0 = d[d.orientation == 0]
        logger.info("%-5s %5d fights  %s -> %s", s, len(o0), o0.event_date.min().date(), o0.event_date.max().date())
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    # Guard: time order between splits.
    assert train.event_date.max() < valid.event_date.min() and valid.event_date.max() < test.event_date.min()

    miss = missingness_report(train, [c for c in numeric if c.startswith("f1_")])
    if len(miss):
        logger.info("Missingness vs target on train (fighter_1 features):\n%s", _fmt_table(miss))
    suspicious = miss[(miss.target_rate_missing - 0.5).abs() > 0.25]
    phys_cols = {f"f1_{p}" for p in PHYSICAL}
    for r in suspicious.itertuples():
        handled = " (handled: imputed before modelling)" if r.feature in phys_cols else ""
        logger.warning("Missing %s coincides with target rate %.2f%s", r.feature, r.target_rate_missing, handled)

    logger.info("Tuning logistic regression on validation")
    logistic = tune_logistic(train, valid, diff_cols, mcfg["logistic"]["C_grid"], seed)
    logger.info("Tuning LightGBM on validation")
    lgbm = tune_lightgbm(train, valid, numeric, categorical, mcfg, seed)
    models = [BaselineModel().fit(train), logistic, lgbm]

    results, calib = {}, {}
    for m in models:
        results[m.name] = {}
        for s in ("valid", "test"):
            flv = fight_level(splits[s], m.predict_proba(splits[s]))
            results[m.name][s] = evaluate(flv)
            calib[(m.name, s)] = calibration_table(flv, mcfg["calibration_bins"])

    rows = [{"model": name, "split": s, **r[s]} for name, r in results.items() for s in ("valid", "test")]
    comparison = pd.DataFrame(rows)[["model", "split", "n_fights", "log_loss", "brier", "accuracy"]]
    logger.info("Model comparison (per fight, symmetric probabilities):\n%s", _fmt_table(comparison))
    for (name, s), t in calib.items():
        if s == "test":
            logger.info("Calibration, %s on test (favourite's predicted p vs actual win rate):\n%s",
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

    # Production model: best validation log loss among the trained models, refit on all data.
    chosen = min((logistic, lgbm), key=lambda m: results[m.name]["valid"]["log_loss"])
    all_rows = pd.concat([train, valid, test], ignore_index=True)
    if chosen is logistic:
        prod = LogisticModel(diff_cols, logistic.C, seed).fit(all_rows)
    else:
        prod = LGBMModel(numeric, categorical, lgbm.params_in, lgbm.best_iteration_).fit(all_rows)
    logger.info("Production model: %s %s, refit on %d fights up to %s", chosen.name, prod.params(),
                all_rows.fight_id.nunique(), all_rows.event_date.max().date())

    models_dir = resolve_path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
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
        "split": cfg["split"],
        "evaluation": {"note": "models fit on train, tuned on valid; production model refit on all data",
                       "metrics": results,
                       "calibration_test": {n: calib[(n, "test")].astype({"bin": str}).to_dict("records")
                                            for n in results}},
        "selection": "lowest validation log loss among logistic and lightgbm",
        "features": {"numeric": numeric if chosen is lgbm else diff_cols,
                     "categorical": categorical if chosen is lgbm else []},
        "importance_flags": flags,
        "lightgbm_top_importance": imp.head(20).round(4).to_dict(),
    }
    artifact = {"model": prod, "metadata": metadata}
    pkl = models_dir / f"model_{today}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(artifact, f)
    (models_dir / f"model_{today}.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    # Copies rather than symlinks (symlinks need admin rights on Windows).
    shutil.copyfile(pkl, models_dir / "latest.pkl")
    shutil.copyfile(models_dir / f"model_{today}.json", models_dir / "latest.json")
    logger.info("Saved %s and models/latest.pkl", pkl)
    return metadata


def load_model(path: Path | None = None) -> dict:
    path = path or resolve_path(load_config()["paths"]["models_dir"]) / "latest.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run_train()
    except Exception:
        logger.exception("Training failed")
        return 1
    return 0


if __name__ == "__main__":
    # Run via the importable module so pickled model classes are stored as `src.train.*`,
    # not `__main__.*` (which could not be unpickled by predict.py).
    from src import train as _train_module
    sys.exit(_train_module.main())
