import numpy as np
import pandas as pd
import pytest

from src.features import build_history, build_training_features, symmetric_probability
from src.predict import OUTPUT_COLUMNS, card_pairs, format_table, predict_card, slugify
from src.train import LogisticModel, fight_level
from src.upcoming import Bout, Card
from tests.test_features import SPLIT, synthetic_tables


@pytest.fixture(scope="module")
def tables():
    return synthetic_tables()


@pytest.fixture(scope="module")
def artifact(tables):
    feats = build_training_features(tables, SPLIT)
    cols = ["diff_n_fights", "diff_win_rate", "diff_streak", "diff_reach_in", "diff_age"]
    model = LogisticModel(cols, 1.0, 0).fit(feats)
    return {"model": model, "metadata": {"model_version": "model_test"}}


def card(date, bouts):
    return Card(event_name="UFC Test: A vs. B", event_date=date, source="manual",
                bouts=[Bout(fighter_1=f1, fighter_2=f2, fighter_1_id=i1, fighter_2_id=i2, weight_class=wc)
                       for f1, i1, f2, i2, wc in bouts])


def test_output_schema_and_probabilities(tables, artifact):
    c = card("2030-01-01", [("A", "a", "B", "b", "Lightweight"), ("E", "e", "New Guy", None, "Welterweight")])
    pred = predict_card(c, tables, artifact, predicted_at="2030-01-01T00:00:00+00:00")
    assert list(pred.columns[:len(OUTPUT_COLUMNS)]) == OUTPUT_COLUMNS
    assert np.allclose(pred.p_fighter_1 + pred.p_fighter_2, 1.0)
    assert (pred.confidence >= 0.5).all()
    assert pred.predicted_winner.tolist() == np.where(pred.p_fighter_1 >= 0.5, pred.fighter_1, pred.fighter_2).tolist()
    assert pred.fighter_2_debut.tolist() == [0, 1] and pred.fighter_1_debut.tolist() == [0, 0]
    assert pred.bout_order.tolist() == [1, 2]
    assert "New Guy *" in format_table(pred)


def test_swapping_fighters_flips_probability(tables, artifact):
    ab = predict_card(card("2030-01-01", [("A", "a", "B", "b", "Lightweight")]), tables, artifact)
    ba = predict_card(card("2030-01-01", [("B", "b", "A", "a", "Lightweight")]), tables, artifact)
    assert ab.p_fighter_1.iloc[0] == pytest.approx(ba.p_fighter_2.iloc[0], abs=1e-4)


def test_same_code_path_as_training(tables, artifact):
    """Predicting a historical fight 'as a card' gives exactly the training-path probability."""
    feats = build_training_features(tables, SPLIT)
    fl = fight_level(feats, artifact["model"].predict_proba(feats)).set_index("fight_id")
    f = tables["fights"].set_index("fight_id").loc["f4"]  # a vs d, 2008-01-01
    c = card(str(f.event_date.date()), [("A", f.fighter_1_id, "D", f.fighter_2_id, f.weight_class)])
    c.bouts[0].scheduled_rounds, c.bouts[0].is_title_fight = int(f.scheduled_rounds), int(f.is_title_fight)
    pred = predict_card(c, tables, artifact)
    assert pred.p_fighter_1.iloc[0] == pytest.approx(fl.loc["f4", "p"], abs=1e-4)


def test_defaults_for_rounds_and_title():
    c = card("2030-01-01", [("A", "a", "B", "b", "Lightweight"), ("C", "c", "D", "d", "Lightweight")])
    p = card_pairs(c)
    assert p.scheduled_rounds.tolist() == [5.0, 3.0] and p.is_title_fight.tolist() == [0.0, 0.0]


def test_slugify():
    assert slugify("UFC Fight Night: Tsarukyan vs. Van") == "ufc-fight-night-tsarukyan-vs-van"
    assert slugify("Noche UFC: Silva vs. Delgado") == "noche-ufc-silva-vs-delgado"
