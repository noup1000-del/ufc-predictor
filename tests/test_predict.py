import numpy as np
import pandas as pd
import pytest

from src.features import build_history, build_training_features, symmetric_probability
from src.predict import EXTRA_COLUMNS, OUTPUT_COLUMNS, card_pairs, format_table, predict_card, slugify
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


@pytest.fixture(scope="module")
def lgbm_artifact(tables):
    from src.features import CATEGORICAL, numeric_feature_columns
    from src.train import LGBMModel
    feats = build_training_features(tables, SPLIT)
    params = {"num_leaves": 4, "min_child_samples": 2, "min_data_in_bin": 1, "learning_rate": 0.3, "random_state": 0}
    model = LGBMModel(numeric_feature_columns(), CATEGORICAL, params, 20).fit(feats)
    return {"model": model, "metadata": {"model_version": "model_lgbm_test", "model_type": "lightgbm",
                                         "data_cutoff": "2011-01-01"}}


def test_contributions_sum_to_model_log_odds():
    from tests.test_train import CATEGORICAL, NUMERIC, make_rows
    from src.train import LGBMModel
    train, test = make_rows(400, 0), make_rows(50, 1)
    logit = lambda p: np.log(p / (1 - p))
    for m in (LogisticModel([c for c in NUMERIC if c.startswith("diff_")], 1.0, 0).fit(train),
              LGBMModel(NUMERIC, CATEGORICAL, {"num_leaves": 7, "random_state": 0}, 50).fit(train)):
        c = m.contributions(test)
        assert "_bias" in c.columns
        assert np.allclose(c.sum(axis=1), logit(m.predict_proba(test)), atol=1e-6), m.name


@pytest.mark.parametrize("which", ["artifact", "lgbm_artifact"])
def test_combined_contributions_mirror_prediction_sign(tables, request, which):
    from src.features import matchup_features, swap_orientation
    from src.predict import explain
    art = request.getfixturevalue(which)
    c = card("2030-01-01", [("A", "a", "B", "b", "Lightweight"), ("B", "b", "A", "a", "Lightweight"),
                            ("C", "c", "D", "d", "Lightweight"), ("E", "e", "New Guy", None, "Welterweight")])
    pairs, history = card_pairs(c), build_history(tables)
    ab, ba = matchup_features(pairs, history), matchup_features(swap_orientation(pairs), history)
    m = art["model"]
    p_ab, p_ba = m.predict_proba(ab), m.predict_proba(ba)
    total = explain(m, ab, ba).sum(axis=1).to_numpy()
    logit = lambda p: np.log(p / (1 - p))
    assert np.allclose(total, 0.5 * (logit(p_ab) - logit(p_ba)), atol=1e-6)
    p1 = symmetric_probability(p_ab, p_ba)
    assert (np.sign(total) == np.sign(np.round(p1 - 0.5, 12))).all()
    assert total[0] == pytest.approx(-total[1])  # swapping the fighters flips every driver


def test_key_factors_and_drivers(tables, lgbm_artifact):
    c = card("2030-01-01", [("A", "a", "B", "b", "Lightweight"), ("E", "e", "New Guy", None, "Welterweight")])
    pred = predict_card(c, tables, lgbm_artifact)
    assert pred.key_factors.str.contains(r"^A: .* \| B: ", regex=True).iloc[0]
    for d in pred["_drivers"]:
        assert len(d["fighter_1"]) <= 3 and len(d["fighter_2"]) <= 3
        assert all(x["logodds"] > 0 for x in d["fighter_1"]) and all(x["logodds"] < 0 for x in d["fighter_2"])
        assert all(" vs " in x["values"] for x in d["fighter_1"] + d["fighter_2"])


def test_html_report_structure(tables, lgbm_artifact, tmp_path):
    from src.report import render_html
    c = card("2030-01-04", [("Raúl <Rosas> Jr.", "a", "O'Brien & Co", "b", "Bantamweight"),
                            ("E", "e", "New Guy", None, "Welterweight")])
    pred = predict_card(c, tables, lgbm_artifact, predicted_at="2030-01-01T00:00:00+00:00")
    path = tmp_path / "card.html"
    path.write_text(render_html(pred, c, lgbm_artifact["metadata"]), encoding="utf-8")
    html = path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "UFC Test: A vs. B" in html and "model_lgbm_test" in html and "Friday 04 January 2030" in html
    assert html.count('<article class="bout">') == 2
    assert html.count('class="bar"') == 2 and html.count('badge pick') == 2
    assert html.count("badge debut") == 1 and "Not found in UFC data" in html
    assert "Main event" in html and "Favours" in html
    assert "Raúl &lt;Rosas&gt; Jr." in html and "O&#x27;Brien &amp; Co" in html   # escaped
    for external in ("http://", "https://", "<script", "<link", "@import", "url("):
        assert external not in html   # fully self-contained


def test_slugify():
    assert slugify("UFC Fight Night: Tsarukyan vs. Van") == "ufc-fight-night-tsarukyan-vs-van"
    assert slugify("Noche UFC: Silva vs. Delgado") == "noche-ufc-silva-vs-delgado"
