"""ufc.com schedule: events-list and event-page parsing, card files, sync, aliases."""
import json
from datetime import date
from pathlib import Path

import pytest

from src.http import FetchError
from src.matching import FighterMatcher
from src.upcoming import (Bout, Card, ScheduledEvent, UpcomingCardError, card_filename, clean_event_name,
                          fetch_ufc_schedule, load_aliases, load_manual_card, match_card, parse_ufc_event_page,
                          parse_ufc_events_page, save_cards)
from tests.test_upcoming import fighters

FIX = Path(__file__).parent / "fixtures"
BASE = "https://www.ufc.com"


def events_html():
    return (FIX / "ufc_events.synthetic.html").read_text(encoding="utf-8")


def event_html():
    return (FIX / "ufc_event.synthetic.html").read_text(encoding="utf-8")


def test_events_page_upcoming_only_sorted_and_deduplicated():
    events = parse_ufc_events_page(events_html(), BASE)
    assert [e.url for e in events] == [f"{BASE}/event/ufc-340", f"{BASE}/event/road-to-ufc-season-6-finals",
                                       f"{BASE}/event/ufc-fight-night-january-02-2027"]  # past + undated skipped
    assert events[0].event_date == "2026-12-12"
    assert events[0].location == "T-Mobile Arena, Las Vegas, NV, United States"
    assert events[0].headline == "Champ vs Challenger"


def test_event_date_uses_displayed_day_and_timestamp_year():
    # 9 PM EST on Jan 2 is Jan 3 in UTC: the displayed (Eastern) day wins, the year comes from the timestamp
    assert parse_ufc_events_page(events_html(), BASE)[-1].event_date == "2027-01-02"


def test_events_page_without_upcoming_list_fails_clearly():
    with pytest.raises(UpcomingCardError):
        parse_ufc_events_page("<html><body>maintenance</body></html>", BASE)


def test_event_page_bouts():
    ev = ScheduledEvent(url=f"{BASE}/event/ufc-340", headline="Champ vs Challenger", event_date="2026-12-12",
                        location="Somewhere")
    card = parse_ufc_event_page(event_html(), ev)
    assert card.event_name == "UFC 340: Champ vs Challenger"   # sponsor prefix dropped
    assert card.source == "ufc.com" and card.event_url == ev.url and card.location == "Somewhere"
    got = [(b.bout_order, b.fighter_1, b.fighter_2, b.weight_class, b.is_title_fight, b.scheduled_rounds)
           for b in card.bouts]
    assert got == [
        (1, "Champ Person", "Chall Enger", "Lightweight", 1, 5),
        (2, "Ana Two", "Bea Three", "Women's Flyweight", 1, 5),   # title bout: 5 rounds
        (3, "Co Main", "Mononym", "Middleweight", 0, 3),          # single-name fighter kept
        (4, "Cat Weight", "Other Guy", "Catch Weight", 0, 3),
        (5, "Early Bird", "Last Fight", "Flyweight", 0, 3),       # bout without an opponent skipped
    ]


def test_clean_event_name():
    assert clean_event_name("Polymarket UFC 334: Gane vs Hokit") == "UFC 334: Gane vs Hokit"
    assert clean_event_name("UFC Fight Night | Bonfim vs Brady") == "UFC Fight Night: Bonfim vs Brady"
    assert clean_event_name("Noche UFC: A vs B") == "Noche UFC: A vs B"
    assert clean_event_name("UFC 333: Volkanovski vs Evloev") == "UFC 333: Volkanovski vs Evloev"


class FakeClient:
    def __init__(self, pages, fail=()):
        self.pages, self.fail, self.calls = pages, set(fail), []

    def fetch(self, page_type, page_id, url, refresh=False):
        self.calls.append((url, refresh))
        if url in self.fail:
            raise FetchError(url, "HTTP 403")
        return self.pages[url]


CFG = {"upcoming": {"ufc_events_url": f"{BASE}/events", "ufc_base_url": BASE}}


def test_fetch_schedule_skips_non_ufc_and_failed_pages():
    client = FakeClient({f"{BASE}/events": events_html(), f"{BASE}/event/ufc-340": event_html()},
                        fail={f"{BASE}/event/ufc-fight-night-january-02-2027"})
    cards = fetch_ufc_schedule(client, CFG, today=date(2026, 12, 1))
    assert [c.event_name for c in cards] == ["UFC 340: Champ vs Challenger"]
    assert all(refresh for _, refresh in client.calls)   # the schedule is always re-fetched
    assert f"{BASE}/event/road-to-ufc-season-6-finals" not in [u for u, _ in client.calls]


def test_fetch_schedule_skips_past_events():
    client = FakeClient({f"{BASE}/events": events_html()})
    assert fetch_ufc_schedule(client, CFG, today=date(2027, 1, 3)) == []


def test_fetch_schedule_refused_raises():
    with pytest.raises(FetchError):
        fetch_ufc_schedule(FakeClient({}, fail={f"{BASE}/events"}), CFG, today=date(2026, 12, 1))


def ufc_card(name, day, url, bouts=1):
    return Card(event_name=name, event_date=day, source="ufc.com", event_url=url, location="Vegas",
                bouts=[Bout(f"A{i}", f"B{i}", "Lightweight", i + 1, is_title_fight=0, scheduled_rounds=3)
                       for i in range(bouts)])


def write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_save_cards_writes_replaces_and_syncs(tmp_path):
    cards_dir, manual = tmp_path / "cards", tmp_path / "upcoming_card.json"
    today = date(2026, 12, 1)
    # A hand-made override is moved to a backup, never silently lost.
    write(manual, {"event_name": "Mine", "event_date": "2026-12-05", "bouts": []})
    save_cards([ufc_card("UFC 340: X vs Y", "2026-12-12", "u340"), ufc_card("UFC 341: Gone", "2026-12-19", "u341")],
               cards_dir, manual, today, fetched_at="t1")
    backups = list(tmp_path.glob("upcoming_card.manual-*.json"))
    assert len(backups) == 1 and json.loads(backups[0].read_text(encoding="utf-8"))["event_name"] == "Mine"
    hand = cards_dir / "2026-12-26_my-own-card.json"
    write(hand, {"event_name": "Own", "event_date": "2026-12-26", "bouts": []})
    write(cards_dir / "2026-11-01_ufc-339-old.json",
          {"event_name": "Old", "event_date": "2026-11-01", "source": "ufc.com", "event_url": "u339", "bouts": []})

    # Headliner changed for u340; u341 is no longer on the schedule.
    second = [ufc_card("UFC 340: X vs Z", "2026-12-12", "u340", bouts=2)]
    save_cards(second, cards_dir, manual, today, fetched_at="t2")
    assert sorted(p.name for p in cards_dir.glob("*.json")) == [
        "2026-11-01_ufc-339-old.json", "2026-12-12_ufc-340-x-vs-z.json", hand.name]  # past + hand-made kept
    synced = json.loads(manual.read_text(encoding="utf-8"))
    assert synced["source"] == "ufc.com" and synced["event_name"] == "UFC 340: X vs Z" and len(synced["bouts"]) == 2
    assert len(list(tmp_path.glob("upcoming_card.manual-*.json"))) == 1   # a synced file is not backed up

    card = load_manual_card(cards_dir / card_filename(second[0]), today)
    assert card.source == "ufc.com" and card.location == "Vegas" and card.event_url == "u340"
    assert [(b.bout_order, b.scheduled_rounds, b.is_title_fight) for b in card.bouts] == [(1, 3, 0), (2, 3, 0)]


def test_save_cards_skips_sync_when_no_card_has_bouts(tmp_path):
    manual = tmp_path / "upcoming_card.json"
    save_cards([ufc_card("UFC 340: TBA", "2026-12-12", "u340", bouts=0)], tmp_path / "cards", manual,
               date(2026, 12, 1))
    assert not manual.exists()


def test_save_cards_is_idempotent(tmp_path):
    cards_dir, manual = tmp_path / "cards", tmp_path / "upcoming_card.json"
    cards = [ufc_card("UFC 340: X vs Y", "2026-12-12", "u340")]
    save_cards(cards, cards_dir, manual, date(2026, 12, 1), fetched_at="t")
    before = {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")}
    save_cards(cards, cards_dir, manual, date(2026, 12, 1), fetched_at="t")
    assert {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")} == before


def test_aliases_map_card_names_and_ignore_unknown_ids(tmp_path):
    path = tmp_path / "aliases.csv"
    path.write_text("name,fighter_id,note\nJosé  Alias,van,verified\nGhost,nope,typo\n", encoding="utf-8")
    aliases = load_aliases(path, known_ids={"van", "pantoja"})
    assert aliases == {"jose alias": "van"}
    card = Card("UFC X", "2099-01-01", "ufc.com", [Bout("Jose Alias", "Alexandre Pantoja", "Flyweight", 1)])
    b = match_card(card, FighterMatcher.from_fighters(fighters()), aliases).bouts[0]
    assert (b.fighter_1_id, b.fighter_1_match) == ("van", "alias")
    assert (b.fighter_2_id, b.fighter_2_match) == ("pantoja", "exact")


def test_committed_aliases_file_is_well_formed():
    path = Path(__file__).parents[1] / "upcoming_aliases.csv"
    aliases = load_aliases(path)
    assert aliases and all(len(fid) == 16 for fid in aliases.values())


# --------------------------------------------------------------------------- line-up changes

from src.upcoming import describe_change, diff_bouts  # noqa: E402


def test_diff_bouts_replacement_added_removed():
    old = [("Raul Rosas Jr.", "Raoni Barcelos"), ("Mickey Gall", "Sedriques Dumas"), ("A One", "B Two")]
    new = [("Raoni Barcelos", "Raúl Rosas Jr"),              # corner swap + spelling: not a change
           ("Luis Hernandez", "Sedriques Dumas"),            # Gall out, Hernandez in
           ("Kyle Nelson", "Cristian Perez Gonzalez")]       # new bout; A One vs B Two is gone
    change = diff_bouts(old, new)
    assert change == {"replaced": [{"out": "Mickey Gall", "in": "Luis Hernandez", "opponent": "Sedriques Dumas"}],
                      "added": [["Kyle Nelson", "Cristian Perez Gonzalez"]],
                      "removed": [["A One", "B Two"]]}
    assert describe_change(change) == ("Luis Hernandez replaces Mickey Gall (vs Sedriques Dumas); "
                                       "added Kyle Nelson vs Cristian Perez Gonzalez; removed A One vs B Two")
    assert diff_bouts(old, old) is None


def test_save_cards_records_changes_and_keeps_history(tmp_path):
    cards_dir, manual = tmp_path / "cards", tmp_path / "upcoming_card.json"
    today = date(2026, 12, 1)
    v1 = ufc_card("UFC 340: X vs Y", "2026-12-12", "u340", bouts=2)
    save_cards([v1], cards_dir, manual, today, fetched_at="2026-12-01T10:00:00+00:00")
    assert load_manual_card(cards_dir / card_filename(v1), today).changes == []   # first fetch: no change

    v2 = ufc_card("UFC 340: X vs Y", "2026-12-12", "u340", bouts=2)
    v2.bouts[1] = Bout("A1", "New Guy", "Lightweight", 2)                     # B1 replaced by New Guy
    save_cards([v2], cards_dir, manual, today, fetched_at="2026-12-02T10:00:00+00:00")
    v3 = ufc_card("UFC 340: X vs Z", "2026-12-12", "u340", bouts=1)            # headliner renamed, bout 2 gone
    save_cards([v3], cards_dir, manual, today, fetched_at="2026-12-03T10:00:00+00:00")
    save_cards([v3], cards_dir, manual, today, fetched_at="2026-12-04T10:00:00+00:00")   # unchanged

    card = load_manual_card(cards_dir / card_filename(v3), today)
    assert [c["detected_at"][:10] for c in card.changes] == ["2026-12-02", "2026-12-03"]
    assert card.changes[0]["replaced"] == [{"out": "B1", "in": "New Guy", "opponent": "A1"}]
    assert card.changes[1]["removed"] == [["A1", "New Guy"]]
    assert card.fetched_at == "2026-12-04T10:00:00+00:00"
    assert json.loads(manual.read_text(encoding="utf-8"))["changes"] == card.changes   # synced card too
