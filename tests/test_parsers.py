import math

import pytest

from src.ingest import (extract_id, fight_duration, parse_date, parse_height, parse_int, parse_mss,
                        parse_of, parse_reach, parse_time_format, parse_weight)

MISSING = ["--", "---", "", "  ", None, float("nan")]


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def test_parse_of():
    assert parse_of("12 of 30") == (12, 30)
    assert parse_of(" 0 of 0 ") == (0, 0)
    for m in MISSING:
        assert all(isnan(v) for v in parse_of(m))
    with pytest.raises(ValueError):
        parse_of("12/30")


def test_parse_mss():
    assert parse_mss("4:32") == 272
    assert parse_mss("0:00") == 0
    assert parse_mss("15:00") == 900
    for m in MISSING:
        assert isnan(parse_mss(m))
    with pytest.raises(ValueError):
        parse_mss("4.32")


def test_parse_height():
    assert parse_height("5' 11\"") == 71
    assert parse_height("6' 0\"") == 72
    assert parse_height("5'11\"") == 71
    for m in MISSING:
        assert isnan(parse_height(m))
    with pytest.raises(ValueError):
        parse_height("180 cm")


def test_parse_reach():
    assert parse_reach('72"') == 72
    assert parse_reach('70.5"') == 70.5
    for m in MISSING:
        assert isnan(parse_reach(m))


def test_parse_weight():
    assert parse_weight("155 lbs.") == 155
    assert parse_weight("265 lbs") == 265
    for m in MISSING:
        assert isnan(parse_weight(m))


def test_parse_int_accepts_float_formatted_ints():
    assert parse_int("3") == 3
    assert parse_int("2.0") == 2
    assert isnan(parse_int("--"))
    with pytest.raises(ValueError):
        parse_int("2.5")


def test_parse_date():
    assert parse_date("September 19, 2026") == "2026-09-19"
    assert parse_date("Jul 13, 1978") == "1978-07-13"
    assert parse_date("Jul. 12, 2025") == "2025-07-12"
    assert parse_date("--") is None
    with pytest.raises(ValueError):
        parse_date("2025-07-12x")


def test_extract_id():
    assert extract_id("http://ufcstats.com/fight-details/568ec6af4008355a") == "568ec6af4008355a"
    assert extract_id("http://ufcstats.com/event-details/8a0a35e7c74bebcc/") == "8a0a35e7c74bebcc"
    assert extract_id("http://ufcstats.com/fighter-details/93fe7332d16c6ad9") == "93fe7332d16c6ad9"
    assert extract_id("") is None


def test_time_format_and_duration():
    assert parse_time_format("3 Rnd (5-5-5)") == (3, [300, 300, 300])
    assert parse_time_format("5 Rnd (5-5-5-5-5)")[0] == 5
    assert parse_time_format("1 Rnd + OT (12-3)") == (2, [720, 180])
    sched, lengths = parse_time_format("No Time Limit")
    assert isnan(sched) and lengths is None

    assert fight_duration(3, 300, [300] * 3) == 900
    assert fight_duration(2, 105, [300] * 3) == 405
    assert fight_duration(1, 296, [300] * 5) == 296
    assert fight_duration(1, 500, None) == 500          # no time limit, ended in "round 1"
    assert isnan(fight_duration(2, 60, None))           # unknown round lengths
    assert isnan(fight_duration(4, 60, [300] * 3))      # inconsistent
