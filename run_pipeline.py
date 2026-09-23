"""Run the UFC predictor pipeline.

    python run_pipeline.py [--update]        ingest new -> clean -> features -> track -> train (if needed) -> predict
    python run_pipeline.py --full            rebuild from the cached source (no downloads), always retrain
    python run_pipeline.py --predict-only
    python run_pipeline.py --stage features  one stage (ingest|clean|features|track|train|predict|backtest)
    python run_pipeline.py --backtest        walk-forward backtest -> models/backtest_report.json
    python run_pipeline.py --dry-run         show the plan without running anything

Options: --card PATH (card JSON for predict), --force-train, --no-predict (stop after train).
Exit code 0 on success, 1 if any stage fails (later stages are not run).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

logger = logging.getLogger("pipeline")

STAGES = ["ingest", "clean", "features", "track", "train", "predict"]
# Validation findings that are known properties of the source, not failures.
INGEST_NONFATAL_CHECKS = {"fights with 0 stat rows (no stats in source)"}


class StageError(Exception):
    """A stage finished but its output failed validation."""


@dataclass
class Context:
    full: bool = False
    force_train: bool = False
    card: Path | None = None


# --------------------------------------------------------------------------- stages

def stage_ingest(ctx: Context) -> dict:
    from src import ingest
    manifest = ingest.run_ingest(full=ctx.full)
    report = ingest.validate_raw(resolve_path(load_config()["paths"]["raw_dir"]))
    ingest.log_validation(report)
    bad = [c for c in report["checks"] if c["violations"] and c["check"] not in INGEST_NONFATAL_CHECKS]
    if bad:
        raise StageError(f"raw validation failed: {[c['check'] for c in bad]}")
    return {"source_commit": manifest["commit_sha"][:10], "added": manifest["added_rows"],
            "raw_rows": {k: v[0] for k, v in report["shapes"].items()},
            "dropped_fights": manifest["dropped_fights"]}


def stage_clean(ctx: Context) -> dict:
    from src import clean
    report = clean.run_clean()
    if clean.log_report(report):
        raise StageError("clean validation failed (see error-level checks above)")
    return {t: c["processed"] for t, c in report["row_counts"].items()}


def stage_features(ctx: Context) -> dict:
    from src import features
    out = features.run_features()
    return {"rows": len(out), "fights": int(out["fight_id"].nunique()),
            "by_split": out[out.orientation == 0].groupby("split").size().to_dict()}


def stage_track(ctx: Context) -> dict:
    from src import track
    r = track.run_track()
    m = (r.get("metrics") or {}).get("overall") or {}
    return {"new_rows": r["added"], "log_rows": r.get("log_rows", 0), "tracked_fights": m.get("fights", 0),
            "accuracy": m.get("accuracy")}


def training_needed(cfg: dict) -> tuple[bool, str]:
    models_dir = resolve_path(cfg["paths"]["models_dir"])
    meta_path = models_dir / "latest.json"
    if not (models_dir / "latest.pkl").exists() or not meta_path.exists():
        return True, "no saved model"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fights_path = resolve_path(cfg["paths"]["processed_dir"]) / "fights.csv"
    if fights_path.exists():
        latest = pd.read_csv(fights_path, usecols=["event_date"])["event_date"].max()
        if str(latest) > meta["data_cutoff"]:
            return True, f"new fights up to {latest} (model data cutoff {meta['data_cutoff']})"
    fl_path = resolve_path(cfg["paths"]["features_dir"]) / "feature_list.json"
    if fl_path.exists():
        current = json.loads(fl_path.read_text(encoding="utf-8"))
        if not set(meta["features"]["numeric"]) <= set(current["numeric"]) or \
                not set(meta["features"]["categorical"]) <= set(current["categorical"]):
            return True, "feature list changed"
    return False, f"model is up to date (data cutoff {meta['data_cutoff']})"


def stage_train(ctx: Context) -> dict:
    needed, reason = training_needed(load_config())
    if not (needed or ctx.force_train):
        logger.info("Skipping training: %s", reason)
        return {"skipped": reason}
    logger.info("Training: %s", "forced" if ctx.force_train and not needed else reason)
    from src import train
    meta = train.run_train()
    test = meta["evaluation"]["metrics"][meta["model_type"]]["test"]
    return {"model_version": meta["model_version"], "model_type": meta["model_type"],
            "fights_trained": meta["n_fights_trained"],
            "test_log_loss": round(test["probabilistic"]["log_loss"], 4),
            "test_auc": round(test["discrimination"]["roc_auc"], 4),
            "test_ece": round(test["calibration"]["ece"], 4),
            "promoted": meta["promotion"]["promoted"]}


def stage_backtest(ctx: Context) -> dict:
    from src import train
    report = train.run_backtest_cli(save=True)
    return {"slices": len(report["slices"]),
            "lightgbm_log_loss": [round(s["metrics"]["lightgbm"]["probabilistic"]["log_loss"], 4)
                                  for s in report["slices"]]}


def stage_predict(ctx: Context) -> dict:
    from src import predict
    pred, path = predict.run_predict(ctx.card)
    return {"bouts": len(pred), "file": path.name,
            "debuts": int(pred["fighter_1_debut"].sum() + pred["fighter_2_debut"].sum())}


STAGE_FUNCS = {"ingest": stage_ingest, "clean": stage_clean, "features": stage_features,
               "track": stage_track, "train": stage_train, "predict": stage_predict,
               "backtest": stage_backtest}  # backtest also runs inside train (promotion gate)


# --------------------------------------------------------------------------- orchestration

def plan(args: argparse.Namespace) -> tuple[list[str], Context]:
    ctx = Context(full=args.full, force_train=args.force_train or args.full, card=args.card)
    if args.predict_only:
        return ["predict"], ctx
    if args.backtest:
        return ["backtest"], ctx
    if args.stage:
        if args.stage == "train":
            ctx.force_train = True  # asking for the train stage means train
        return [args.stage], ctx
    stages = list(STAGES)
    if args.no_predict:
        stages.remove("predict")
    return stages, ctx


def run(stages: list[str], ctx: Context) -> int:
    t0 = time.perf_counter()
    logger.info("Pipeline start: %s", " -> ".join(stages))
    for name in stages:
        s0 = time.perf_counter()
        logger.info("=== %s: start", name)
        try:
            summary = STAGE_FUNCS[name](ctx)
        except Exception as e:  # noqa: BLE001 - report any stage failure and stop with exit 1
            logger.exception("=== %s: FAILED after %.1fs: %s", name, time.perf_counter() - s0, e)
            logger.error("Pipeline stopped at stage %r (%.1fs total)", name, time.perf_counter() - t0)
            return 1
        logger.info("=== %s: done in %.1fs %s", name, time.perf_counter() - s0, summary)
    logger.info("Pipeline finished in %.1fs", time.perf_counter() - t0)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="default mode")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--predict-only", action="store_true")
    mode.add_argument("--backtest", action="store_true", help="walk-forward backtest only")
    mode.add_argument("--stage", choices=STAGES + ["backtest"])
    p.add_argument("--card", type=Path, help="card JSON for the predict stage")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--no-predict", action="store_true",
                   help="with --update/--full: stop after train (e.g. a post-event results/tracking job)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def setup_logging() -> None:
    log_dir = resolve_path("logs")
    log_dir.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(), logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging()
    stages, ctx = plan(args)
    if args.dry_run:
        logger.info("Dry run. Stages: %s | full=%s force_train=%s card=%s", " -> ".join(stages),
                    ctx.full, ctx.force_train, ctx.card)
        if "train" in stages:
            needed, reason = training_needed(load_config())
            logger.info("Train stage would %s: %s", "run" if needed or ctx.force_train else "be skipped",
                        "forced" if ctx.force_train and not needed else reason)
        return 0
    return run(stages, ctx)


if __name__ == "__main__":
    sys.exit(main())
