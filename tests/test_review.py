"""Post-event review: scorecard joins, lessons, evidence board thresholds, Results tab."""
import json

import numpy as np
import pandas as pd
import pytest

from src.review import (MIN_EVIDENCE_FIGHTS, bout_records, build_reviews, event_lessons, evidence_board,
                        parse_key_factors, surprise_level)
from src.track import load_predictions


def test_surprise_levels():
    assert surprise_level(0.51) == "coin flip"
    assert surprise_level(0.60) == "lean"
    assert surprise_level(0.65) == surprise_level(0.9) == "clear favourite"


def test_parse_key_factors_strips_weights_and_splits_sides():
    text = "Raul Rosas Jr.: Age 22.0 vs 39.4 (0.35); Win rate 50% vs 36% (0.26) | Raoni Barcelos: none"
    assert parse_key_factors(text, "Raul Rosas Jr.", "Raoni Barcelos") == {
        "fighter_1": ["Age 22.0 vs 39.4", "Win rate 50% vs 36%"], "fighter_2": []}
    assert parse_key_factors(None, "A", "B") == {"fighter_1": [], "fighter_2": []}


# --------------------------------------------------------------------------- a small tracked event

FIGHTS = pd.DataFrame({
    "fight_id": ["f1", "f2", "f3", "f4"], "event_date": ["2030-01-05"] * 4,
    "fighter_1_id": ["a", "c", "e", "g"], "fighter_2_id": ["b", "d", "f", "h"],
    "winner_id": ["a", "d", "f", pd.NA], "result": ["win", "win", "win", "draw"],
    "method": ["KO/TKO", "Decision - Unanimous", "Submission", "Decision - Split"],
    "method_group": ["ko_tko", "decision", "submission", "decision"],
    "end_round": [1, 3, 2, 3], "end_time_sec": [100, 300, 200, 300], "scheduled_rounds": [5, 3, 3, 3],
    "weight_class": ["Lightweight", "Women's Flyweight", "Welterweight", "Bantamweight"],
}).astype({"winner_id": "string"})


def write_event(tmp_path):
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    pred = pd.DataFrame({
        "event_date": ["2030-01-05"] * 4, "weight_class": FIGHTS.weight_class,
        "fighter_1": ["Al A", "Cy C", "Ed E", "Gus G"], "fighter_2": ["Bo B", "Di D", "Fi F", "Hal H"],
        "p_fighter_1": [0.7, 0.8, 0.52, 0.6], "p_fighter_2": [0.3, 0.2, 0.48, 0.4],
        "predicted_winner": ["Al A", "Cy C", "Ed E", "Gus G"], "confidence": [0.7, 0.8, 0.52, 0.6],
        "model_version": "m1", "predicted_at": "2030-01-02T00:00:00+00:00", "event_name": "UFC Test: A vs B",
        "bout_order": [1, 2, 3, 4], "fighter_1_id": ["a", "c", "e", "g"], "fighter_2_id": ["b", "d", None, "h"],
        "fighter_1_match": "exact", "fighter_2_match": ["exact", "exact", "unmatched", "exact"],
        "fighter_1_debut": [0, 0, 0, 0], "fighter_2_debut": [0, 0, 1, 0],
        "key_factors": ["Al A: Age 25.0 vs 35.0 (0.30) | Bo B: none",
                        "Cy C: Win rate 80% vs 40% (0.50); Reach 70\" vs 65\" (0.10) | Di D: Age 30.0 vs 24.0 (0.05)",
                        "Ed E: none | Fi F: none", "Gus G: none | Hal H: none"],
    })
    pred.to_csv(pred_dir / "2030-01-05_ufc-test-a-vs-b.csv", index=False)
    log = pd.DataFrame({
        "event_date": ["2030-01-05"] * 4, "fighter_1": pred.fighter_1, "fighter_2": pred.fighter_2,
        "predicted_winner": pred.predicted_winner, "actual_winner": ["Al A", "Di D", "Fi F", "draw"],
        "correct": ["True", "False", "False", None], "p_predicted_winner": ["0.7", "0.8", "0.52", "0.6"],
        "model_version": "m1", "tracked_at": "t", "fight_id": ["f1", "f2", "f3", "f4"],
        "event_name": "UFC Test: A vs B", "prediction_file": "2030-01-05_ufc-test-a-vs-b.csv"})
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "2030-01-05_ufc-test-a-vs-b.json").write_text(json.dumps({
        "event_name": "UFC Test: A vs B", "event_date": "2030-01-05", "source": "ufc.com", "bouts": [],
        "changes": [{"detected_at": "2030-01-03T10:00:00+00:00", "added": [],
                     "replaced": [{"out": "Old Guy", "in": "Di D", "opponent": "Cy C"}], "removed": []}]}),
        encoding="utf-8")
    return load_predictions(pred_dir), log, cards


def test_bout_records_join_by_ids_and_flag_debut_and_late_change(tmp_path):
    preds, log, cards = write_event(tmp_path)
    b = bout_records(log, FIGHTS, preds, cards).set_index("fight_id")
    assert b.loc["f1", "correct"] is True and b.loc["f2", "correct"] is False and b.loc["f4", "correct"] is None
    assert b.loc["f2", "surprise"] == "clear favourite" and b.loc["f3", "surprise"] == "coin flip"
    assert b.loc["f3", "debut"] and b.loc["f3", "debutants"] == ["Fi F"]          # found via the ids of the pair
    assert b.loc["f2", "late_change"] and b.loc["f2", "late_change_detected_at"].startswith("2030-01-03")
    assert not b.loc["f1", "late_change"]
    assert b.loc["f2", "pick_factors"] == ["Win rate 80% vs 40%", 'Reach 70" vs 65"']
    assert b.loc["f2", "other_factors"] == ["Age 30.0 vs 24.0"]
    assert (b.loc["f1", "method_group"], b.loc["f1", "end_round"], b.loc["f1", "scheduled_rounds"]) == ("ko_tko", 1, 5)


def test_event_lessons_are_rule_based_and_cautious(tmp_path):
    preds, log, cards = write_event(tmp_path)
    lessons = event_lessons(bout_records(log, FIGHTS, preds, cards))
    text = "\n".join(lessons)
    assert lessons[0].startswith("1 of 3 picks correct (33%)") and "expected about 2.0 correct" in lessons[0]
    assert "Upset: Di D beat Cy C (80% for Cy C) by decision" in text
    assert "Win rate 80% vs 40%" in text and "(values: Cy C vs Di D)" in text
    assert "near coin flips" in text and "Fi F over Ed E" in text
    assert "Bouts with a late line-up change: 0 of 1 correct" in text
    assert "1 bout(s) ended in a draw or no contest" in text
    assert lessons[-1].startswith("One card is not evidence")


def test_evidence_board_needs_enough_fights_and_a_clear_gap():
    rng = np.random.default_rng(0)
    n = 200
    b = pd.DataFrame({"p_pick": np.full(n, 0.7), "correct": [True] * n, "debut": [False] * n,
                      "late_change": [False] * n, "method_group": ["decision"] * n,
                      "scheduled_rounds": [3] * n, "weight_class": ["Lightweight"] * n})
    b.loc[:139, "correct"] = rng.random(140) < 0.7                  # 140 normal fights: ~70% as expected
    b.loc[140:, "debut"] = True
    b.loc[140:, "correct"] = [True] * 18 + [False] * 42            # 60 debut fights: 30% vs 70% expected
    board = evidence_board(b)
    seg = {(g["segment"], g["value"]): g for g in board["segments"]}
    assert seg[("UFC debutant in the bout", "yes")]["status"] == "overconfident"
    assert seg[("UFC debutant in the bout", "no")]["status"] == "consistent"
    small = evidence_board(b.iloc[140:140 + MIN_EVIDENCE_FIGHTS - 1])    # same pattern, too few fights
    assert {g["status"] for g in small["segments"]} == {"collecting"}


def test_reviews_and_results_tab(tmp_path):
    from src.report import render_index
    preds, log, cards = write_event(tmp_path)
    reviews, board = build_reviews(bout_records(log, FIGHTS, preds, cards), "2030-01-06T00:00:00+00:00")
    assert len(reviews) == 1 and reviews[0]["slug"] == "ufc-test-a-vs-b"
    assert reviews[0]["summary"]["fights"] == 3 and reviews[0]["summary"]["correct"] == 1
    json.dumps(reviews)   # serialisable
    html = render_index([], {"model_version": "m"}, generated="2030-01-06T00:00:00+00:00",
                        reviews=reviews, board=board)
    panel = html[html.index('id="results"'):html.index('id="card-changes"')]
    assert "Tracked picks <b>1 of 3</b> (33%)" in panel and "Results &amp; lessons learned" in panel
    assert panel.count('<span class="res ok">Hit</span>') == 1 and panel.count('<span class="res miss">Miss</span>') == 2
    assert "Late change" in panel and "Debut" in panel and "KO/TKO, R1" in panel
    assert "Evidence board" in panel and "collecting evidence" in panel
    empty = render_index([], {"model_version": "m"}, generated="2030-01-06T00:00:00+00:00")
    assert "No finished events tracked yet." in empty and "Results appear here" in empty
