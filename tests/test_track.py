import numpy as np
import pandas as pd
import pytest

from src.track import append_log, build_log_rows, match_predictions, running_metrics

TRACKED_AT = "2026-10-10T00:00:00+00:00"


def fights():
    return pd.DataFrame([
        # fight_id, date, f1, f2, winner, result
        ("F1", "2026-10-03", "a", "b", "a", "win"),
        ("F2", "2026-10-03", "c", "d", None, "draw"),
        ("F3", "2026-10-03", "e", "x", "x", "win"),     # x debuted; unknown at prediction time
        ("F4", "2026-09-01", "a", "c", "c", "win"),     # an older fight of a
    ], columns=["fight_id", "event_date", "fighter_1_id", "fighter_2_id", "winner_id", "result"])


def pred(f1, i1, f2, i2, p1, date="2026-10-03", predicted_at="2026-10-01T12:00:00+00:00", file="card.csv"):
    winner = f1 if p1 >= 0.5 else f2
    return {"event_date": date, "fighter_1": f1, "fighter_2": f2, "fighter_1_id": i1, "fighter_2_id": i2,
            "predicted_winner": winner, "confidence": str(max(p1, 1 - p1)), "p_fighter_1": str(p1),
            "model_version": "model_x", "predicted_at": predicted_at, "event_name": "UFC Test",
            "prediction_file": file}


def preds():
    return pd.DataFrame([
        pred("B", "b", "A", "a", 0.3),              # card order swapped vs source; picks A
        pred("C", "c", "D", "d", 0.6),              # draw
        pred("E", "e", "Newbie", None, 0.7),        # debutant without id; E lost
        pred("G", "g", "H", "h", 0.5),              # not fought yet
    ])


def test_matching_by_id_pair_regardless_of_order():
    m = match_predictions(preds(), fights())
    assert m.fight_id.tolist()[:3] == ["F1", "F2", "F3"]
    assert m.status.tolist() == ["matched", "matched", "matched", "pending"]


def test_date_tolerance_and_mismatch():
    p = pd.DataFrame([pred("A", "a", "B", "b", 0.6, date="2026-10-04"),   # UTC date one day later
                      pred("A", "a", "Z", "z", 0.6),                     # a fought b, not z
                      pred("A", "a", "B", "b", 0.6, date="2026-10-10")])  # too far from the fight
    m = match_predictions(p, fights())
    assert m.status.tolist() == ["matched", "pending", "pending"]
    assert m.fight_id.iloc[0] == "F1"


def test_log_rows_winner_correct_and_draws():
    rows = build_log_rows(match_predictions(preds(), fights()), fights(), TRACKED_AT).set_index("fight_id")
    assert rows.loc["F1", "actual_winner"] == "A" and rows.loc["F1", "correct"] == True  # noqa: E712
    assert rows.loc["F1", "p_predicted_winner"] == pytest.approx(0.7)
    assert rows.loc["F2", "actual_winner"] == "draw" and pd.isna(rows.loc["F2", "correct"])
    assert rows.loc["F3", "actual_winner"] == "Newbie" and rows.loc["F3", "correct"] == False  # noqa: E712
    assert "F4" not in rows.index


def test_late_predictions_are_not_counted():
    p = pd.DataFrame([pred("A", "a", "B", "b", 0.6, predicted_at="2026-10-09T00:00:00+00:00")])
    assert build_log_rows(match_predictions(p, fights()), fights(), TRACKED_AT).empty


def test_latest_prediction_wins_when_duplicated():
    p = pd.DataFrame([pred("A", "a", "B", "b", 0.4, predicted_at="2026-09-30T00:00:00+00:00", file="old.csv"),
                      pred("A", "a", "B", "b", 0.8, predicted_at="2026-10-02T00:00:00+00:00", file="new.csv")])
    rows = build_log_rows(match_predictions(p, fights()), fights(), TRACKED_AT)
    assert len(rows) == 1 and rows.prediction_file.iloc[0] == "new.csv" and rows.correct.iloc[0] == True  # noqa: E712


def test_append_is_idempotent(tmp_path):
    path = tmp_path / "results_log.csv"
    rows = build_log_rows(match_predictions(preds(), fights()), fights(), TRACKED_AT)
    assert append_log(path, rows) == 3
    assert append_log(path, rows) == 0
    later = build_log_rows(match_predictions(preds(), fights()), fights(), "2030-01-01T00:00:00+00:00")
    assert append_log(path, later) == 0
    log = pd.read_csv(path, dtype=str)
    assert len(log) == 3 and set(log.tracked_at) == {TRACKED_AT}   # existing rows untouched


def test_running_metrics():
    log = pd.DataFrame({
        "event_date": ["2026-01-01"] * 2 + ["2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-06-01"],
        "correct": ["True", "False", "True", "True", "False", "True", "True", ""],
        "p_predicted_winner": ["0.6", "0.7", "0.8", "0.55", "0.65", "0.9", "0.6", "0.5"],
    })
    m = running_metrics(log, recent_events=5)
    y = np.array([1, 0, 1, 1, 0, 1, 1]); p = np.array([0.6, 0.7, 0.8, 0.55, 0.65, 0.9, 0.6])
    assert m["overall"]["fights"] == 7 and m["overall"]["accuracy"] == pytest.approx(y.mean())
    assert m["overall"]["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["last_5_events"]["fights"] == 5          # 2026-01-01 falls out of the window
    assert m["excluded_draw_nc"] == 1 and m["events_overall"] == 6
