import pandas as pd
import pytest

from src.clean import (SCHEMA, _to_csv_frame, apply_types, clean_tables, load_processed, method_group)

TS = "2026-09-23T00:00:00+00:00"


def fight(fid, date, f1, f2, result="win", winner="f1", method="KO/TKO", eid=None):
    return {"fight_id": fid, "event_id": eid or f"e{date}", "event_date": date, "bout_order": "1",
            "fighter_1_id": f1, "fighter_2_id": f2, "fighter_1_name": f1.upper(), "fighter_2_name": f2.upper(),
            "winner_id": {"f1": f1, "f2": f2}.get(winner) if result == "win" else None,
            "result": result, "method": method, "end_round": "1", "end_time_sec": "100",
            "scheduled_rounds": "3", "weight_class": "Lightweight", "is_title_fight": "0"}


def stat(fid, fighter, opp, landed="10"):
    row = {c: "1" for c in SCHEMA["fight_stats"]}
    row.update({"fight_id": fid, "fighter_id": fighter, "opponent_id": opp, "sig_str_landed": landed,
                "sig_str_attempted": "20", "fight_duration_sec": "100", "ctrl_sec": None})
    return row


def raw_tables():
    fights = [
        fight("old", "2004-06-01", "a", "b"),                      # before cutoff
        fight("w1", "2006-01-01", "a", "b"),
        fight("d1", "2007-01-01", "a", "c", result="draw", method="Decision - Split"),
        fight("n1", "2008-01-01", "b", "c", result="nc", method="Overturned"),
        fight("one", "2009-01-01", "a", "c", winner="f2", method="Submission"),  # only one stat row
    ]
    stats = [stat("old", "a", "b"), stat("old", "b", "a"),
             stat("w1", "a", "b"), stat("w1", "b", "a"),
             stat("d1", "a", "c"), stat("d1", "c", "a"),
             stat("n1", "b", "c"), stat("n1", "c", "b"),
             stat("one", "a", "c")]
    events = [{"event_id": f["event_id"], "event_name": f["fight_id"], "event_date": f["event_date"],
               "location": "X", "scraped_at": TS} for f in fights]
    fighters = [{"fighter_id": i, "name": i, "nickname": None, "height_in": "70.0", "weight_lb": "155",
                 "reach_in": None, "stance": "Orthodox", "dob": "1985-01-01", "scraped_at": TS}
                for i in "abc"]
    return {"fights": pd.DataFrame(fights), "fight_stats": pd.DataFrame(stats),
            "events": pd.DataFrame(events), "fighters": pd.DataFrame(fighters)}


@pytest.fixture
def cleaned():
    return clean_tables(raw_tables(), "2005-01-01")


def test_cutoff_drops_fights_stats_and_events(cleaned):
    t, report = cleaned
    assert "old" not in set(t["fights"].fight_id)
    assert "old" not in set(t["fight_stats"].fight_id)
    assert "e2004-06-01" not in set(t["events"].event_id)
    assert t["fights"].event_date.min() >= pd.Timestamp("2005-01-01")
    assert report["row_counts"]["fights"] == {"raw": 5, "processed": 4}
    assert {"table": "fights", "reason": "before min_date 2005-01-01 (moved to fights_prior)", "rows": 1} \
        in report["dropped"]


def test_pre_cutoff_fights_kept_as_history_only(cleaned):
    t, report = cleaned
    prior = t["fights_prior"]
    assert list(prior.fight_id) == ["old"]
    assert not prior.is_target.any() and not prior.has_stats.any()
    assert prior.winner_id.iloc[0] == "a"
    assert report["row_counts"]["fights_prior"] == {"raw": None, "processed": 1}


def test_draws_and_ncs_kept_but_not_targets(cleaned):
    t, report = cleaned
    f = t["fights"].set_index("fight_id")
    assert {"d1", "n1"} <= set(f.index)
    assert not f.loc["d1", "is_target"] and not f.loc["n1", "is_target"]
    assert f.loc["w1", "is_target"] and f.loc["one", "is_target"]
    assert report["targets"] == {"is_target": 2, "excluded_draw": 1, "excluded_nc": 1}


def test_fight_with_single_stat_row_kept_without_stats(cleaned):
    t, report = cleaned
    f = t["fights"].set_index("fight_id")
    assert "one" in f.index and not f.loc["one", "has_stats"]
    assert "one" not in set(t["fight_stats"].fight_id)
    assert t["fight_stats"].groupby("fight_id").size().eq(2).all()
    errors = [c for c in report["checks"] if c["severity"] == "error" and c["violations"]]
    assert errors == []


def test_types(cleaned):
    t, _ = cleaned
    f, s, fr = t["fights"], t["fight_stats"], t["fighters"]
    assert str(f.event_date.dtype).startswith("datetime64")
    assert str(f.bout_order.dtype) == "Int64" and str(f.is_target.dtype) == "boolean"
    assert str(s.sig_str_landed.dtype) == "Int64" and s.ctrl_sec.isna().all()
    assert fr.reach_in.isna().all() and fr.height_in.iloc[0] == 70.0
    assert str(fr.scraped_at.dtype).endswith("UTC]")


def test_method_group():
    assert method_group("KO/TKO") == "ko_tko"
    assert method_group("TKO - Doctor's Stoppage") == "ko_tko"
    assert method_group("Submission") == "submission"
    assert method_group("Decision - Majority") == "decision"
    assert method_group("DQ") == "dq"
    assert method_group("Overturned") == "other"
    assert method_group(None) is None


def test_parse_failures_reported():
    raw = raw_tables()
    raw["fights"].loc[1, "event_date"] = "2006-13-45"
    raw["fighters"].loc[0, "dob"] = "not a date"
    t, report = clean_tables(raw, "2005-01-01")
    assert "w1" not in set(t["fights"].fight_id)
    fails = {(p["table"], p["column"]): p["rows"] for p in report["parse_failures"]}
    assert fails == {("fights", "event_date"): 1, ("fighters", "dob"): 1}
    assert {"table": "fights", "reason": "event_date missing or unparseable", "rows": 1} in report["dropped"]


def test_duplicates_removed_and_reported():
    raw = raw_tables()
    raw["fights"] = pd.concat([raw["fights"], raw["fights"].iloc[[1]]], ignore_index=True)
    t, report = clean_tables(raw, "2005-01-01")
    assert t["fights"].fight_id.is_unique
    assert {"table": "fights", "reason": "duplicate key", "rows": 1} in report["dropped"]


def test_csv_round_trip_preserves_types(cleaned, tmp_path):
    t, _ = cleaned
    for name, df in t.items():
        _to_csv_frame(name, df).to_csv(tmp_path / f"{name}.csv", index=False)
    back = load_processed(tmp_path)
    for name in t:
        pd.testing.assert_frame_equal(back[name].reset_index(drop=True), t[name].reset_index(drop=True),
                                      check_categorical=False)
