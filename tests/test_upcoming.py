import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from src.matching import FighterMatcher
from src.upcoming import load_manual_card, match_card, parse_espn_scoreboard

FIX = Path(__file__).parent / "fixtures"


def fighters():
    return pd.DataFrame({
        "fighter_id": ["van", "pantoja", "bruno_fly", "bruno_mid", "turman", "davis", "burns",
                       "vologdin", "moura", "oneill", "tsarukyan"],
        "name": ["Joshua Van", "Alexandre Pantoja", "Bruno Silva", "Bruno Silva", "Wellington Turman",
                 "Mike Davis", "Gilbert Burns", "Mark Vologdin", "Eduarda Moura", "Casey O'Neill",
                 "Arman Tsarukyan"],
        "weight_lb": [125, 125, 125, 185, 185, 155, 170, 135, 125, 125, 155],
        "dob": [None] * 11,
    })


def test_espn_picks_next_ufc_card_and_orders_competitors():
    data = json.loads((FIX / "espn_scoreboard.synthetic.json").read_text(encoding="utf-8"))
    card = parse_espn_scoreboard(data, today=date(2026, 9, 23))
    # Skips DWCS (earlier) and the completed UFC 331; picks Oct 3 over Oct 10.
    assert card.event_name == "UFC Fight Night: Tsarukyan vs. Van"
    assert card.event_date == "2026-10-03" and card.source == "espn"
    assert len(card.bouts) == 2  # the one-competitor competition is skipped
    b = card.bouts[0]
    assert (b.fighter_1, b.fighter_2, b.weight_class) == ("Casey O'Neill", "Eduarda Moura", "Women's Flyweight")
    assert card.bouts[1].fighter_2 == "Joshua Vann"  # fullName fallback


def test_espn_returns_none_when_nothing_upcoming():
    data = json.loads((FIX / "espn_scoreboard.synthetic.json").read_text(encoding="utf-8"))
    assert parse_espn_scoreboard(data, today=date(2026, 12, 1)) is None


def test_espn_card_matching_uses_fuzzy_fallback():
    data = json.loads((FIX / "espn_scoreboard.synthetic.json").read_text(encoding="utf-8"))
    card = match_card(parse_espn_scoreboard(data, today=date(2026, 9, 23)),
                      FighterMatcher.from_fighters(fighters()))
    b = card.bouts[1]
    assert (b.fighter_1_id, b.fighter_1_match) == ("tsarukyan", "exact")
    assert (b.fighter_2_id, b.fighter_2_match) == ("van", "fuzzy")
    assert card.unmatched() == []


def test_manual_card(tmp_path):
    p = tmp_path / "upcoming_card.json"
    shutil.copy(FIX / "upcoming_card.example.json", p)
    card = load_manual_card(p, today=date(2026, 9, 23))
    assert card.source == "manual" and card.event_date == "2099-01-01"
    assert card.bouts[2].fighter_1_id == "fb3e61720be4690c" and card.bouts[2].fighter_1_match == "manual"

    card = match_card(card, FighterMatcher.from_fighters(fighters()))
    assert card.bouts[0].fighter_1_id == "van"
    assert card.bouts[1].fighter_1_id == "bruno_mid"          # weight class disambiguates
    assert card.bouts[2].fighter_1_id == "fb3e61720be4690c"   # manual id kept
    assert card.unmatched() == ["Totally New Debutant"]


def test_manual_card_in_the_past_is_ignored(tmp_path):
    p = tmp_path / "upcoming_card.json"
    p.write_text(json.dumps({"event_date": "2020-01-01", "bouts": []}), encoding="utf-8")
    assert load_manual_card(p, today=date(2026, 9, 23)) is None
