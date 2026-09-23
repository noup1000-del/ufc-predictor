import json

import pandas as pd
import pytest

import run_pipeline as rp


@pytest.fixture
def recorder(monkeypatch):
    calls = []

    def make(name, fail=False):
        def stage(ctx):
            calls.append((name, ctx.full, ctx.force_train))
            if fail:
                raise RuntimeError(f"{name} broke")
            return {"ok": True}
        return stage

    monkeypatch.setattr(rp, "STAGE_FUNCS", {n: make(n) for n in rp.STAGES})
    monkeypatch.setattr(rp, "setup_logging", lambda: None)
    return calls, make, monkeypatch


def test_update_runs_all_stages_in_order(recorder):
    calls, _, _ = recorder
    assert rp.main([]) == 0
    assert [c[0] for c in calls] == ["ingest", "clean", "features", "track", "train", "predict"]
    assert all(not full and not force for _, full, force in calls)


def test_full_mode_rebuilds_and_forces_training(recorder):
    calls, _, _ = recorder
    assert rp.main(["--full"]) == 0
    assert len(calls) == 6 and all(full and force for _, full, force in calls)


def test_predict_only_and_single_stage(recorder):
    calls, _, _ = recorder
    assert rp.main(["--predict-only"]) == 0
    assert rp.main(["--stage", "features"]) == 0
    assert rp.main(["--stage", "train"]) == 0
    assert [c[0] for c in calls] == ["predict", "features", "train"]
    assert calls[2][2] is True  # explicit train stage forces training


def test_no_predict_stops_after_train(recorder):
    calls, _, _ = recorder
    assert rp.main(["--update", "--no-predict"]) == 0
    assert [c[0] for c in calls] == ["ingest", "clean", "features", "track", "train"]


def test_failure_stops_pipeline_with_nonzero_exit(recorder):
    calls, make, monkeypatch = recorder
    funcs = dict(rp.STAGE_FUNCS)
    funcs["features"] = make("features", fail=True)
    monkeypatch.setattr(rp, "STAGE_FUNCS", funcs)
    assert rp.main(["--update"]) == 1
    assert [c[0] for c in calls] == ["ingest", "clean", "features"]


def test_dry_run_executes_nothing(recorder):
    calls, _, monkeypatch = recorder
    monkeypatch.setattr(rp, "training_needed", lambda cfg: (False, "up to date"))
    assert rp.main(["--dry-run"]) == 0
    assert calls == []


def test_invalid_stage_is_rejected(recorder):
    with pytest.raises(SystemExit) as e:
        rp.main(["--stage", "nope"])
    assert e.value.code == 2


def test_train_stage_skips_when_model_is_current(monkeypatch):
    monkeypatch.setattr(rp, "training_needed", lambda cfg: (False, "model is up to date"))
    import src.train
    monkeypatch.setattr(src.train, "run_train", lambda: pytest.fail("should not train"))
    assert rp.stage_train(rp.Context()) == {"skipped": "model is up to date"}


def _cfg(tmp_path):
    for d in ("models", "processed", "features"):
        (tmp_path / d).mkdir()
    return {"paths": {"models_dir": str(tmp_path / "models"), "processed_dir": str(tmp_path / "processed"),
                      "features_dir": str(tmp_path / "features")}}


def test_training_needed_rules(tmp_path):
    cfg = _cfg(tmp_path)
    assert rp.training_needed(cfg) == (True, "no saved model")

    (tmp_path / "models" / "latest.pkl").write_bytes(b"x")
    meta = {"data_cutoff": "2026-09-19", "features": {"numeric": ["a", "b"], "categorical": ["c"]}}
    (tmp_path / "models" / "latest.json").write_text(json.dumps(meta))
    pd.DataFrame({"event_date": ["2026-09-12", "2026-09-19"]}).to_csv(tmp_path / "processed" / "fights.csv")
    (tmp_path / "features" / "feature_list.json").write_text(json.dumps({"numeric": ["a", "b", "z"], "categorical": ["c"]}))
    needed, reason = rp.training_needed(cfg)
    assert not needed and "up to date" in reason

    pd.DataFrame({"event_date": ["2026-09-19", "2026-09-26"]}).to_csv(tmp_path / "processed" / "fights.csv")
    needed, reason = rp.training_needed(cfg)
    assert needed and "2026-09-26" in reason

    pd.DataFrame({"event_date": ["2026-09-19"]}).to_csv(tmp_path / "processed" / "fights.csv")
    (tmp_path / "features" / "feature_list.json").write_text(json.dumps({"numeric": ["a"], "categorical": ["c"]}))
    assert rp.training_needed(cfg) == (True, "feature list changed")
