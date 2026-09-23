from datetime import date

from src.matching import (FighterInfo, FighterMatcher, canonical_weight_class, normalize_name,
                          weight_limit_lb)


def test_normalize_name():
    assert normalize_name("José Aldo") == "jose aldo"
    assert normalize_name("Casey O'Neill") == "casey oneill"
    assert normalize_name("  Michael  Aswell Jr. ") == "michael aswell jr"
    assert normalize_name("Jan Błachowicz") == "jan blachowicz"
    assert normalize_name("Mairbek Taisumov") == normalize_name("MAIRBEK  TAISUMOV")
    assert normalize_name(None) == ""


def test_canonical_weight_class():
    assert canonical_weight_class("UFC Flyweight Title Bout") == "Flyweight"
    assert canonical_weight_class("UFC Women's Strawweight Title Bout") == "Women's Strawweight"
    assert canonical_weight_class("Light Heavyweight Bout") == "Light Heavyweight"
    assert canonical_weight_class("Heavyweight Bout") == "Heavyweight"
    assert canonical_weight_class("Catch Weight Bout") == "Catch Weight"
    assert canonical_weight_class("UFC 2 Tournament Title Bout") is None
    assert weight_limit_lb("Women's Bantamweight") == 135
    assert weight_limit_lb("Light Heavyweight") == 205
    assert weight_limit_lb("Catch Weight") is None


def make_matcher():
    names = [("fly", "Bruno Silva"), ("mid", "Bruno Silva"), ("van", "Joshua Van"),
             ("van", "Josh Van"), ("dav1", "Mike Davis"), ("dav2", "Mike Davis"),
             ("yoo", "JooSang Yoo"), ("old", "Jean Silva"), ("new", "Jean Silva")]
    info = {"fly": FighterInfo(125, date(1990, 3, 16)), "mid": FighterInfo(185, date(1989, 7, 13)),
            "dav1": FighterInfo(None, None), "dav2": FighterInfo(155, date(1992, 10, 7)),
            "old": FighterInfo(160, date(1950, 1, 1)), "new": FighterInfo(145, date(1996, 12, 27))}
    return FighterMatcher(names, info)


def test_exact_unique_and_alias():
    m = make_matcher()
    assert m.match("Joshua Van").fighter_id == "van"
    assert m.match("JOSH VAN").fighter_id == "van"
    assert m.match("Joshua Van").method == "exact"


def test_same_name_disambiguated_by_weight():
    m = make_matcher()
    r = m.match("Bruno Silva", weight_class="Flyweight", on_date=date(2021, 1, 1))
    assert (r.fighter_id, r.method) == ("fly", "exact_disambiguated")
    assert m.match("Bruno Silva", weight_class="Middleweight Bout").fighter_id == "mid"
    # Bantamweight (135): fly is 10 lb away, mid 50 lb -> clear.
    assert m.match("Bruno Silva", weight_class="Bantamweight").fighter_id == "fly"


def test_same_name_disambiguated_by_age():
    m = make_matcher()
    # "old" would be 76 at the fight -> excluded, even with no weight class.
    assert m.match("Jean Silva", on_date=date(2026, 1, 1)).fighter_id == "new"


def test_ambiguous_when_evidence_unclear():
    m = make_matcher()
    r = m.match("Mike Davis", weight_class="Lightweight", on_date=date(2020, 1, 1))
    assert r.fighter_id is None and r.method == "ambiguous"
    assert set(r.candidates) == {"dav1", "dav2"}
    # No weight class and no age evidence for Bruno Silva either.
    assert m.match("Bruno Silva").method == "ambiguous"


def test_no_fuzzy_unless_asked():
    m = make_matcher()
    assert m.match("Joshua Vann").method == "unmatched"
    r = m.match("Joshua Vann", fuzzy=True)
    assert (r.fighter_id, r.method) == ("van", "fuzzy") and r.score >= 0.92


def test_compact_and_reordered_names():
    m = make_matcher()
    assert m.match("Joo Sang Yoo", fuzzy=True).fighter_id == "yoo"
    assert m.match("Van Joshua", fuzzy=True).fighter_id == "van"


def test_fuzzy_rejects_weak_or_close_matches():
    m = make_matcher()
    assert m.match("Totally New Debutant", fuzzy=True).fighter_id is None
    # "Bruno Silvo" is equally close to both Bruno Silvas' shared name -> ambiguous, not guessed.
    assert m.match("Bruno Silvo", fuzzy=True).fighter_id is None
