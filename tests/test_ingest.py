from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.ingest import (SCHEMAS, build_tables, load_overrides, load_source, merge_csv, validate_raw)

FIX = Path(__file__).parent / "fixtures" / "greco"
FILES = ["ufc_event_details", "ufc_fight_results", "ufc_fight_stats", "ufc_fighter_details", "ufc_fighter_tott"]
TODAY = date(2026, 9, 23)
TS = "2026-09-23T00:00:00+00:00"

VAN, PANTOJA = "17e97649403ba428", "a0f0004aadf10b71"
BRUNO_FLY, BRUNO_MID = "294aa73dbf37d281", "12ebd7d157e91701"
MIKE_DAVIS = "fb3e61720be4690c"
TITLE_FIGHT, KO_FIGHT, NC_FIGHT, DRAW_FIGHT = ("568ec6af4008355a", "fe612d20eec7ef35",
                                               "7e8749cbf32ef1f8", "552f7cdaf93e1055")
BRUNO_FLY_FIGHT, BRUNO_MID_FIGHT = "1a2d13834696c4ab", "bb1ac382e0277b3f"
DAVIS_FIGHT, RENAMED_EVENT_FIGHT = "c2bcdcc472228768", "114d2f6fcd3f6f00"


@pytest.fixture(scope="module")
def src():
    return load_source(FIX, FILES)


@pytest.fixture(scope="module")
def built(src):
    return build_tables(src, {}, TODAY, TS)


def fight(tables, fid):
    f = tables["fights"]
    return f[f.fight_id == fid].iloc[0]


def test_outcome_mapping(built):
    tables, _ = built
    t = fight(tables, TITLE_FIGHT)
    assert (t.fighter_1_id, t.fighter_2_id, t.winner_id, t.result) == (VAN, PANTOJA, VAN, "win")
    assert (t.weight_class, t.is_title_fight, t.scheduled_rounds, t.end_round, t.end_time_sec) == \
        ("Flyweight", 1, 5, 5, 300)
    assert t.bout_order == 1
    assert fight(tables, KO_FIGHT).bout_order == 2
    nc = fight(tables, NC_FIGHT)
    assert nc.result == "nc" and pd.isna(nc.winner_id) and nc["_duration"] == 405
    draw = fight(tables, DRAW_FIGHT)
    assert draw.result == "draw" and pd.isna(draw.winner_id)


def test_loser_listed_first(built):
    tables, _ = built
    f = fight(tables, BRUNO_FLY_FIGHT)  # "Bruno Silva vs. David Dvorak", OUTCOME L/W
    assert f.fighter_1_id == BRUNO_FLY and f.winner_id == f.fighter_2_id


def test_round_aggregation_matches_source(built, src):
    tables, _ = built
    st = tables["fight_stats"]
    van = st[(st.fight_id == TITLE_FIGHT) & (st.fighter_id == VAN)].iloc[0]
    # Source rounds: 13+37+65+14+52 of 27+64+98+28+88; ctrl 0:45+0:20+0:14+0:00+0:54.
    assert (van.sig_str_landed, van.sig_str_attempted) == (181, 305)
    assert van.ctrl_sec == 133 and van.kd == 1 and van.reversals == 1
    assert van.opponent_id == PANTOJA and van.fight_duration_sec == 1500
    assert van.sig_head_landed + van.sig_body_landed + van.sig_leg_landed == van.sig_str_landed


def test_renamed_event_not_duplicated(built, src):
    tables, dropped = built
    f = tables["fights"]
    assert (f.fight_id == RENAMED_EVENT_FIGHT).sum() == 1
    st = tables["fight_stats"]
    rows = st[st.fight_id == RENAMED_EVENT_FIGHT]
    assert len(rows) == 2
    # Stats must not be double counted although the source lists the rounds under both event names.
    fs = src["ufc_fight_stats"]
    per_name = fs[(fs.BOUT == "Kelvin Gastelum vs. Dustin Stoltzfus") & (fs.FIGHTER == "Kelvin Gastelum")]
    one_event = per_name[per_name.EVENT == per_name.EVENT.iloc[0]]
    landed = sum(int(x.split(" of ")[0]) for x in one_event["SIG.STR."])
    gastelum = rows[rows.fighter_id == fight(tables, RENAMED_EVENT_FIGHT).fighter_1_id].iloc[0]
    assert gastelum.sig_str_landed == landed
    assert RENAMED_EVENT_FIGHT not in set(dropped.fight_id)


def test_same_name_disambiguation(built):
    tables, _ = built
    assert fight(tables, BRUNO_FLY_FIGHT).fighter_1_id == BRUNO_FLY
    assert fight(tables, BRUNO_MID_FIGHT).fighter_2_id == BRUNO_MID


def test_ambiguous_name_dropped_without_override(built):
    tables, dropped = built
    assert DAVIS_FIGHT not in set(tables["fights"].fight_id)
    reason = dropped.set_index("fight_id").loc[DAVIS_FIGHT, "reason"]
    assert "ambiguous" in reason and "Mike Davis" in reason


def test_override_resolves_ambiguous_name(src, tmp_path):
    p = tmp_path / "ovr.csv"
    p.write_text(f"fight_id,name,fighter_id,note\n{DAVIS_FIGHT},Mike Davis,{MIKE_DAVIS},test\n", encoding="utf-8")
    tables, dropped = build_tables(src, load_overrides(p), TODAY, TS)
    f = fight(tables, DAVIS_FIGHT)
    assert f.fighter_2_id == MIKE_DAVIS
    assert len(tables["fight_stats"].query("fight_id == @DAVIS_FIGHT")) == 2
    assert DAVIS_FIGHT not in set(dropped.fight_id)


def test_future_events_skipped(src):
    tables, _ = build_tables(src, {}, date(2026, 9, 18), TS)
    assert "8a0a35e7c74bebcc" not in set(tables["events"].event_id)
    assert TITLE_FIGHT not in set(tables["fights"].fight_id)


def test_merge_is_idempotent_and_validation_passes(built, tmp_path):
    tables, _ = built
    order = ["fighters", "fights", "fight_stats", "events"]
    first = {n: merge_csv(tmp_path / f"{n}.csv", n, tables[n]) for n in order}
    assert first["fights"] == len(tables["fights"]) and first["fight_stats"] == 2 * first["fights"]
    second = {n: merge_csv(tmp_path / f"{n}.csv", n, tables[n]) for n in order}
    assert second == {n: 0 for n in order}
    for n in order:
        df = pd.read_csv(tmp_path / f"{n}.csv", dtype=str)
        assert list(df.columns) == SCHEMAS[n]
        assert len(df) == first[n]

    report = validate_raw(tmp_path, today=TODAY)
    failures = {c["check"]: c["violations"] for c in report["checks"] if c["violations"]}
    assert failures == {}


def test_merge_only_adds_new_keys(built, tmp_path):
    tables, _ = built
    f = tables["fights"]
    merge_csv(tmp_path / "fights.csv", "fights", f.iloc[:3])
    assert merge_csv(tmp_path / "fights.csv", "fights", f) == len(f) - 3
    assert len(pd.read_csv(tmp_path / "fights.csv")) == len(f)
