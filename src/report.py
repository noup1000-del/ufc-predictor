"""Self-contained HTML reports: one page per predicted card (embedded CSS, no scripts, no
external resources, so it works offline and can be emailed as a single file), and the
multi-event dashboard index.html (same, plus one small inline script for the event tabs)."""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from html import escape
from types import SimpleNamespace

import pandas as pd

from src.upcoming import slugify

CSS = """
:root {
  --bg: #0d1117; --panel: #161b22; --panel-2: #1c232d; --line: #2a3340;
  --text: #e6edf3; --muted: #8b98a8; --f1: #e5534b; --f2: #4493f8;
  --pick: #f2c14e; --debut: #b083f0;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 16px 40px; }
header.top { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 22px; }
.kicker { color: var(--muted); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; }
h1 { font-size: clamp(22px, 4vw, 32px); margin: 6px 0 8px; line-height: 1.15; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; color: var(--muted); font-size: 13px; }
.meta b { color: var(--text); font-weight: 600; }
.grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); }
.bout { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px;
  display: flex; flex-direction: column; gap: 12px; }
.bout-head { display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px;
  text-transform: uppercase; letter-spacing: .08em; }
.bout-head .order { color: var(--text); font-weight: 700; }
.fighter { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 8px;
  background: var(--panel-2); border-left: 4px solid transparent; }
.fighter.f1 { border-left-color: var(--f1); }
.fighter.f2 { border-left-color: var(--f2); }
.fighter.winner { outline: 1px solid var(--pick); }
.fighter .name { font-weight: 650; font-size: 16px; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.fighter .prob { font-variant-numeric: tabular-nums; font-weight: 700; font-size: 17px; }
.fighter.f1 .prob { color: var(--f1); }
.fighter.f2 .prob { color: var(--f2); }
.badge { font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; border-radius: 999px;
  padding: 2px 8px; white-space: nowrap; }
.badge.pick { background: var(--pick); color: #1b1300; }
.badge.debut { border: 1px solid var(--debut); color: var(--debut); }
.note { color: var(--muted); font-size: 12px; margin: -6px 0 0 14px; }
.bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; background: var(--line); }
.bar .seg.f1 { background: var(--f1); }
.bar .seg.f2 { background: var(--f2); }
.drivers { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.drivers h3 { margin: 0 0 6px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em;
  font-weight: 600; }
.drivers ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.tag { background: var(--panel-2); border-radius: 6px; padding: 6px 8px; font-size: 13px; }
.tag .v { display: block; color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.tag .s { display: block; height: 3px; border-radius: 2px; margin-top: 5px; }
.col.f1 .tag .s { background: var(--f1); }
.col.f2 .tag .s { background: var(--f2); }
.none { color: var(--muted); font-size: 13px; }
footer { margin-top: 26px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); padding-top: 14px; }
@media (max-width: 420px) { .drivers { grid-template-columns: 1fr; } }
a { color: inherit; }
.back { display: inline-block; margin-bottom: 10px; color: var(--muted); font-size: 13px; text-decoration: none; }
.back:hover { color: var(--text); }
"""

DASHBOARD_CSS = """
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.appbar { position: sticky; top: 0; z-index: 10; background: rgba(13, 17, 23, .94);
  backdrop-filter: blur(6px); border-bottom: 1px solid var(--line); }
.appbar-inner { max-width: 1180px; margin: 0 auto; padding: 12px 16px 8px; display: flex;
  align-items: center; justify-content: space-between; gap: 12px; }
.brand { display: flex; flex-direction: column; min-width: 0; }
.brand-title { font-weight: 700; font-size: 18px; }
.picker { display: none; flex: 1; max-width: 320px; }
.picker select { width: 100%; background: var(--panel); color: var(--text); border: 1px solid var(--line);
  border-radius: 8px; padding: 8px 10px; font: inherit; font-size: 14px; }
.tabs { max-width: 1180px; margin: 0 auto; padding: 0 16px 10px; display: flex; gap: 8px; overflow-x: auto;
  scrollbar-width: thin; }
.tab { flex: 0 0 auto; display: flex; flex-direction: column; gap: 2px; padding: 7px 12px; border-radius: 10px;
  border: 1px solid var(--line); background: var(--panel); color: var(--muted); text-decoration: none;
  max-width: 230px; }
.tab:hover { color: var(--text); border-color: var(--muted); }
.tab:focus-visible { outline: 2px solid var(--pick); outline-offset: 2px; }
.tab .tab-title { color: var(--text); font-weight: 650; font-size: 14px; white-space: nowrap; }
.tab .tab-sub { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.date-badge { display: inline-block; margin-left: 4px; padding: 0 7px; border-radius: 999px; font-size: 11px;
  font-weight: 700; background: var(--panel-2); color: var(--muted); border: 1px solid var(--line);
  vertical-align: 1px; }
.tab.next .date-badge { border-color: var(--pick); color: var(--pick); }
.tab.active { background: var(--panel-2); border-color: var(--pick); color: var(--text); }
.tab.active .date-badge { background: var(--pick); color: #1b1300; border-color: var(--pick); }
.summary { margin: 0 0 18px; }
.event-panel { scroll-margin-top: 130px; }
html:not(.js) .event-panel + .event-panel { margin-top: 40px; padding-top: 24px; border-top: 1px solid var(--line); }
.panel-head { margin-bottom: 18px; }
.panel-head h2 { font-size: clamp(22px, 4vw, 30px); margin: 0 0 8px; line-height: 1.15; }
.panel-head .warn { color: var(--debut); }
.standalone { color: var(--pick); text-decoration: none; font-weight: 600; }
.standalone:hover { text-decoration: underline; }
.label-next { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #1b1300;
  background: var(--pick); border-radius: 999px; padding: 2px 8px; vertical-align: 5px; margin-left: 10px; }
.empty { color: var(--muted); background: var(--panel); border: 1px dashed var(--line); border-radius: 12px;
  padding: 24px 16px; text-align: center; }
.updates { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 6px 16px; }
.feed-date { margin-left: 8px; color: var(--muted); font-size: 12px; }
.seg-title { margin: 26px 0 12px; font-size: 13px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--pick); display: flex; align-items: center; gap: 10px; }
.seg-title span { color: var(--muted); font-weight: 600; letter-spacing: .04em; text-transform: none; font-size: 12px; }
.seg-title::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.panel-head + .seg-title { margin-top: 6px; }
.review { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px;
  margin-bottom: 16px; }
.review header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }
.review h3 { margin: 0; font-size: 17px; }
.review-score { margin-left: auto; font-size: 13px; color: var(--muted); }
.review h4 { margin: 12px 0 6px; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.lessons { margin: 0 0 12px; padding-left: 18px; font-size: 13px; display: flex; flex-direction: column; gap: 4px; }
.table-wrap { overflow-x: auto; }
table.scorecard, table.evidence { width: 100%; border-collapse: collapse; font-size: 13px; }
.scorecard th, .evidence th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: .06em; padding: 6px 8px; border-bottom: 1px solid var(--line); }
.scorecard td, .evidence td { padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.scorecard .vs, .scorecard .p, .scorecard .method, .why .order { color: var(--muted); }
.scorecard .method { display: block; font-size: 12px; }
.why { margin-top: 4px; font-size: 12px; color: var(--muted); }
.res { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .05em; }
.res.ok { background: #3fb950; color: #04260f; }
.res.miss { background: var(--f1); color: #2b0806; }
.res.none { border: 1px solid var(--line); color: var(--muted); }
.surprise { display: block; margin-top: 3px; font-size: 11px; color: var(--muted); }
.flag { display: inline-block; margin-left: 6px; padding: 0 6px; border-radius: 999px; font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .05em; border: 1px solid; vertical-align: 1px; }
.flag.debut { color: var(--debut); }
.flag.late { color: var(--pick); }
.note { color: var(--muted); font-size: 13px; margin: 0 0 10px; max-width: 900px; }
.evidence tr.ev-overconfident td:last-child, .evidence tr.ev-underconfident td:last-child { color: var(--f1); font-weight: 650; }
.evidence tr.ev-consistent td:last-child { color: #3fb950; }
.evidence tr.ev-collecting td:last-child { color: var(--muted); }
.tab-changes { border-style: dashed; }
.tab-changes.active { border-style: solid; }
.count-badge { display: inline-block; min-width: 18px; margin-left: 4px; padding: 0 6px; border-radius: 999px;
  background: var(--pick); color: #1b1300; font-size: 11px; font-weight: 700; text-align: center; vertical-align: 1px; }
.feed { list-style: none; margin: 0; padding: 0; }
.feed-row { display: grid; grid-template-columns: 190px 1fr; gap: 12px; padding: 8px 0; border-top: 1px solid var(--line); }
.feed-row:first-child { border-top: 0; }
.feed-when { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; padding-top: 2px; }
.feed-event { font-weight: 650; font-size: 14px; text-decoration: none; }
.feed-event:hover { text-decoration: underline; }
.feed-none { color: var(--muted); font-size: 13px; margin: 0; }
.chg-list { list-style: none; margin: 4px 0 0; padding: 0; display: flex; flex-direction: column; gap: 4px; font-size: 13px; }
.chg-list s { color: var(--muted); }
.chg { display: inline-block; min-width: 92px; margin-right: 8px; padding: 1px 7px; border-radius: 999px; font-size: 10px;
  font-weight: 700; letter-spacing: .06em; text-transform: uppercase; text-align: center; border: 1px solid; }
.chg.in { color: var(--pick); border-color: var(--pick); }
.chg.add { color: #3fb950; border-color: #3fb950; }
.chg.out { color: var(--f1); border-color: var(--f1); }
.panel-changes { margin-top: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 8px 12px; }
.panel-changes summary { cursor: pointer; font-size: 13px; font-weight: 650; color: var(--pick); }
.panel-changes .feed { margin-top: 6px; }
.upd-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--pick);
  margin-left: 6px; vertical-align: 1px; }
@media (max-width: 720px) {
  .picker { display: block; }
  .tabs { display: none; }
  .brand .kicker { display: none; }
  .brand-title { font-size: 15px; }
  .event-panel { scroll-margin-top: 70px; }
  .feed-row { grid-template-columns: 1fr; gap: 2px; }
  html:not(.js) .picker { display: none; }
  html:not(.js) .tabs { display: flex; }
}
"""


def _bouts(n: int) -> str:
    return f"{n} bout" if n == 1 else f"{n} bouts"


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def _fighter_row(side: str, name: str, p: float, is_pick: bool, conf: float, debut: bool) -> str:
    badges = ""
    if debut:
        badges += '<span class="badge debut">UFC debut</span>'
    if is_pick:
        badges += f'<span class="badge pick">Pick {conf:.0%}</span>'
    cls = f"fighter {side}" + (" winner" if is_pick else "")
    return (f'<div class="{cls}"><span class="name">{escape(name)}</span>{badges}'
            f'<span class="prob">{_pct(p)}</span></div>')


def _match_note(match: str | None, fighter_id) -> str:
    if not isinstance(fighter_id, str) or not fighter_id:
        return '<div class="note">Not found in UFC data; treated as a UFC debut.</div>'
    if match == "fuzzy":
        return '<div class="note">Name matched approximately; check the fighter.</div>'
    return ""


def _drivers_col(side: str, name: str, items: list[dict], scale: float) -> str:
    if items:
        tags = "".join(
            f'<li class="tag" title="{abs(d["logodds"]):.2f} log-odds">{escape(d["label"])}'
            f'<span class="v">{escape(d["values"])}</span>'
            f'<span class="s" style="width:{max(8, 100 * abs(d["logodds"]) / scale):.0f}%"></span></li>'
            for d in items)
        body = f"<ul>{tags}</ul>"
    else:
        body = '<p class="none">No notable factors</p>'
    return f'<div class="col {side}"><h3>Favours {escape(name)}</h3>{body}</div>'


def _bout_card(r) -> str:
    p1, p2 = float(r.p_fighter_1), float(r.p_fighter_2)
    pick1 = r.predicted_winner == r.fighter_1
    drivers = r._drivers or {"fighter_1": [], "fighter_2": []}
    scale = max([abs(d["logodds"]) for side in drivers.values() for d in side] or [1.0])
    order = int(r.bout_order)
    label = "Main event" if order == 1 else f"Bout {order}"
    rounds = f" · {int(r.scheduled_rounds)} rds" if hasattr(r, "scheduled_rounds") else ""
    aria = f"{r.fighter_1} {_pct(p1)}, {r.fighter_2} {_pct(p2)}"
    return f"""
<article class="bout">
  <div class="bout-head"><span class="order">#{order} · {label}</span><span>{escape(str(r.weight_class or ""))}{rounds}</span></div>
  {_fighter_row("f1", r.fighter_1, p1, pick1, float(r.confidence), bool(r.fighter_1_debut))}
  {_match_note(r.fighter_1_match, r.fighter_1_id)}
  <div class="bar" role="img" aria-label="{escape(aria)}"><span class="seg f1" style="width:{100 * p1:.1f}%"></span><span class="seg f2" style="width:{100 * p2:.1f}%"></span></div>
  {_fighter_row("f2", r.fighter_2, p2, not pick1, float(r.confidence), bool(r.fighter_2_debut))}
  {_match_note(r.fighter_2_match, r.fighter_2_id)}
  <div class="drivers">
    {_drivers_col("f1", r.fighter_1, drivers["fighter_1"], scale)}
    {_drivers_col("f2", r.fighter_2, drivers["fighter_2"], scale)}
  </div>
</article>"""


FOOTER_NOTE = ("Probabilities average both fighter orders, so they always sum to 100%. Key factors are the "
               "model's feature contributions (in log-odds, combined over both orders): each tag shows the "
               "values for the fighter on the left vs the right, and the bar shows its size relative to the "
               "strongest factor in that bout. Factors explain this model's estimate, not what will happen "
               "in the fight.")


def _accuracy_note(meta: dict) -> str:
    try:
        acc = meta["evaluation"]["metrics"][meta["model_type"]]["test"]["accuracy"]
    except (KeyError, TypeError):
        return ""
    return f" On held-out test fights this model picked the winner {acc:.0%} of the time."


SEGMENT_TITLES = {"main": "Main card", "prelims": "Prelims", "early_prelims": "Early prelims"}


def _bout_grid(pred: pd.DataFrame) -> str:
    """Bout cards in card order, grouped under Main card / Prelims / Early prelims headings when
    the card says which segment each bout is on (ufc.com cards); one plain grid otherwise."""
    # records, not itertuples: itertuples renames the `_drivers` column
    rows = pred.sort_values("bout_order").to_dict("records")
    segs = [r.get("card_segment") if isinstance(r.get("card_segment"), str) else None for r in rows]

    def grid(items):
        return '<div class="grid">' + "".join(_bout_card(SimpleNamespace(**r)) for r in items) + "</div>"

    if not any(segs):
        return grid(rows)
    groups: list[tuple[str | None, list[dict]]] = []
    for seg, r in zip(segs, rows):
        if not groups or groups[-1][0] != seg:
            groups.append((seg, []))
        groups[-1][1].append(r)
    return "\n".join(f'<h3 class="seg-title">{SEGMENT_TITLES.get(seg, "Other bouts")} '
                     f'<span>{_bouts(len(items))}</span></h3>{grid(items)}' for seg, items in groups)


def render_html(pred: pd.DataFrame, card, meta: dict, index_link: bool = False) -> str:
    """`pred` is predict.predict_card() output (with `_drivers`); `card` has event_name/event_date.
    `index_link` adds a link to this event in index.html (the dashboard, same folder)."""
    d = date.fromisoformat(card.event_date)
    generated = str(pred["predicted_at"].iloc[0]) if len(pred) else ""
    cards = _bout_grid(pred)
    acc_note = _accuracy_note(meta)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(card.event_name)} predictions</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    {f'<a class="back" href="index.html#{event_slug(card.event_name)}">&larr; All upcoming events</a>' if index_link else ""}
    <div class="kicker">Fight predictions</div>
    <h1>{escape(card.event_name)}</h1>
    <div class="meta">
      <span>Date <b>{d.strftime("%A %d %B %Y")}</b></span>{f'''
      <span>Venue <b>{escape(card.location)}</b></span>''' if getattr(card, "location", None) else ""}
      <span>Model <b>{escape(str(meta.get("model_version", "")))}</b> ({escape(str(meta.get("model_type", "")))}, data to {escape(str(meta.get("data_cutoff", "")))})</span>
      <span>Generated <b>{escape(generated)}</b></span>
      <span>{_bouts(len(pred))}</span>
    </div>
  </header>
  <main>{cards}
  </main>
  <footer>
    {FOOTER_NOTE}{acc_note}
  </footer>
</div>
</body>
</html>
"""


def event_slug(event_name: str) -> str:
    """Anchor/id for an event: `UFC 332: Silva vs Wang` -> `ufc-332-silva-vs-wang`."""
    return slugify(event_name) or "event"


def short_title(event_name: str) -> str:
    """Tab label: `UFC 332: Silva vs Wang` -> `UFC 332`; `UFC Fight Night: Allen vs Duncan` ->
    `Allen vs Duncan`; anything else unchanged."""
    m = re.match(r"^((?:Noche )?UFC \d+)\b", event_name)
    if m:
        return m.group(1)
    head, sep, tail = event_name.partition(":")
    return tail.strip() if sep and tail.strip() else event_name


def _headliner_bar(h: dict) -> str:
    p1, p2 = float(h["p_fighter_1"]), float(h["p_fighter_2"])
    aria = f"{h['fighter_1']} {_pct(p1)}, {h['fighter_2']} {_pct(p2)}"
    return (f'<div class="bar" role="img" aria-label="{escape(aria)}"><span class="seg f1" '
            f'style="width:{100 * p1:.1f}%"></span><span class="seg f2" style="width:{100 * p2:.1f}%"></span></div>')


RECENT_CHANGE_DAYS = 7    # tabs get an "updated" dot for changes this recent


def _when(iso: str | None) -> str:
    """'2026-09-26T08:05:02+00:00' -> 'Sat 26 Sep 2026, 08:05 UTC' (source text if unparsable)."""
    if not iso:
        return "unknown"
    try:
        t = datetime.fromisoformat(str(iso)).astimezone(timezone.utc)
    except ValueError:
        return str(iso)
    return t.strftime("%a %d %b %Y, %H:%M UTC")


def _change_items(change: dict) -> str:
    """<li> rows for one detected line-up change (see upcoming.diff_bouts)."""
    rows = [f'<li><span class="chg in">Replacement</span><b>{escape(r["in"])}</b> replaces '
            f'<s>{escape(r["out"])}</s> vs {escape(r["opponent"])}</li>' for r in change.get("replaced", [])]
    rows += [f'<li><span class="chg add">New bout</span><b>{escape(a)}</b> vs <b>{escape(b)}</b></li>'
             for a, b in change.get("added", [])]
    rows += [f'<li><span class="chg out">Off the card</span><s>{escape(a)} vs {escape(b)}</s></li>'
             for a, b in change.get("removed", [])]
    return "".join(rows)


def _is_recent(change: dict, now: datetime) -> bool:
    try:
        t = datetime.fromisoformat(str(change.get("detected_at")))
    except ValueError:
        return False
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return (now - t).days < RECENT_CHANGE_DAYS


CHANGES_ID = "card-changes"   # id/anchor of the all-events "Card changes" tab


def _short_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f'{d.strftime("%a")} {d.day} {d.strftime("%b")}'


def _changes_view(entries: list[dict], slugs: list[str]) -> tuple[str, str, str]:
    """The all-events "Card changes" tab: (tab link, <option>, panel). Every recorded line-up
    change of every scheduled event, newest first, plus when the cards were last checked."""
    items = [(c.get("detected_at") or "", e, s, c) for e, s in zip(entries, slugs) for c in e.get("changes") or []]
    items.sort(key=lambda x: x[0], reverse=True)
    checked = max((e["fetched_at"] for e in entries if e.get("fetched_at")), default=None)
    rows = "".join(
        f'<li class="feed-row"><div class="feed-when">{escape(_when(when))}</div>'
        f'<div><a class="feed-event" href="#{s}">{escape(e["event_name"])}</a>'
        f'<span class="feed-date">{_short_date(e["event_date"])}</span>'
        f'<ul class="chg-list">{_change_items(c)}</ul></div></li>'
        for when, e, s, c in items)
    body = (f'<ul class="feed">{rows}</ul>' if rows else
            '<p class="feed-none">No line-up changes detected since the cards were first fetched.</p>')
    n = len(items)
    count = f'<span class="count-badge">{n}</span>' if n else ""
    tab = (f'<a class="tab tab-changes" role="tab" id="tab-{CHANGES_ID}" href="#{CHANGES_ID}" '
           f'data-target="{CHANGES_ID}" aria-controls="{CHANGES_ID}">'
           f'<span class="tab-title">Card changes {count}</span>'
           f'<span class="tab-sub">All events</span></a>')
    option = f'<option value="{CHANGES_ID}">Card changes ({n})</option>'
    panel = f"""
<section class="event-panel changes-panel" id="{CHANGES_ID}" data-title="Card changes" aria-labelledby="h-{CHANGES_ID}">
  <header class="panel-head">
    <h2 id="h-{CHANGES_ID}">Card changes</h2>
    <div class="meta"><span>Replacements, new bouts and bouts taken off every scheduled card, newest first</span>
      <span>Cards last checked with ufc.com <b>{escape(_when(checked))}</b></span></div>
  </header>
  <div class="updates">{body}</div>
</section>"""
    return tab, option, panel


RESULTS_ID = "results"   # id/anchor of the "Results" tab (post-event reviews + evidence board)
MAX_REVIEWED_EVENTS = 10


def _pct0(v) -> str:
    return "n/a" if v is None else f"{100 * float(v):.0f}%"


def _method_text(b: dict) -> str:
    labels = {"ko_tko": "KO/TKO", "submission": "Submission", "decision": "Decision", "dq": "DQ", "other": "Other"}
    m = labels.get(b.get("method_group") or "", "")
    if not m:
        return ""
    return m if m == "Decision" or not b.get("end_round") else f"{m}, R{b['end_round']}"


def _scorecard(bouts: list[dict]) -> str:
    rows = []
    for b in bouts:
        if b.get("correct") is True:
            mark = '<span class="res ok">Hit</span>'
        elif b.get("correct") is False:
            mark = f'<span class="res miss">Miss</span><span class="surprise">{escape(b.get("surprise") or "")}</span>'
        else:
            mark = f'<span class="res none">{escape((b.get("result") or "").upper() or "n/a")}</span>'
        flags = []
        if b.get("debut"):
            flags.append('<span class="flag debut">Debut</span>')
        if b.get("late_change"):
            flags.append('<span class="flag late">Late change</span>')
        why = ""
        if b.get("correct") is False and b.get("pick_factors"):
            why = (f'<div class="why">Favoured {escape(b["pick"])} on: '
                   f'{escape("; ".join(b["pick_factors"][:3]))} '
                   f'<span class="order">({escape(b["fighter_1"])} vs {escape(b["fighter_2"])})</span></div>')
        rows.append(
            f'<tr><td class="num">{b.get("bout_order") or ""}</td>'
            f'<td>{escape(b["fighter_1"])} <span class="vs">vs</span> {escape(b["fighter_2"])}{"".join(flags)}{why}</td>'
            f'<td><b>{escape(b["pick"])}</b> <span class="p">{_pct0(b.get("p_pick"))}</span></td>'
            f'<td>{escape(str(b.get("actual_winner") or ""))}<span class="method">{escape(_method_text(b))}</span></td>'
            f'<td>{mark}</td></tr>')
    return ('<div class="table-wrap"><table class="scorecard"><thead><tr><th>#</th><th>Bout</th><th>Our pick</th>'
            '<th>Winner</th><th>Result</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def _evidence_table(board: dict) -> str:
    segs = board.get("segments") or []
    if not segs:
        return ""
    rows = "".join(
        f'<tr class="ev-{escape(g["status"])}"><td>{escape(g["segment"])}</td><td>{escape(g["value"])}</td>'
        f'<td class="num">{g["fights"]}</td><td class="num">{_pct0(g["accuracy"])}</td>'
        f'<td class="num">{_pct0(g["expected_accuracy"])}</td>'
        f'<td class="num">{_pct0(g["ci_low"])}&ndash;{_pct0(g["ci_high"])}</td>'
        f'<td>{escape(g["verdict"])}</td></tr>' for g in segs)
    return ('<div class="table-wrap"><table class="evidence"><thead><tr><th>Segment</th><th>Group</th><th>Fights</th>'
            '<th>Hit rate</th><th>Expected</th><th>95% range</th><th>Verdict</th></tr></thead><tbody>'
            + rows + "</tbody></table></div>")


def _results_view(reviews: list[dict] | None, board: dict | None, first_upcoming: str | None) -> tuple[str, str, str]:
    """The "Results" tab: (tab link, <option>, panel) with the post-event reviews (newest first)
    and the evidence board across all tracked fights."""
    reviews = sorted(reviews or [], key=lambda r: r["event_date"], reverse=True)[:MAX_REVIEWED_EVENTS]
    o = (board or {}).get("overall") or {}
    if o.get("fights"):
        record = (f'<span>Tracked picks <b>{o["correct"]} of {o["fights"]}</b> ({_pct0(o["accuracy"])}) over '
                  f'<b>{board.get("events", len(reviews))}</b> event(s)</span>'
                  f'<span>Model expected <b>{o["expected_correct"]:.1f}</b> ({_pct0(o["expected_accuracy"])})</span>'
                  f'<span>Brier <b>{o["brier"]:.3f}</b></span>')
        sub = f'{o["correct"]}/{o["fights"]} picks'
    else:
        record = "<span>No finished events tracked yet.</span>"
        sub = "Coming soon"
    events_html = []
    for r in reviews:
        s = r.get("summary") or {}
        head = (f'{s["correct"]} of {s["fights"]} correct ({_pct0(s["accuracy"])}) &middot; model expected '
                f'{s["expected_correct"]:.1f}' if s.get("fights") else "Not scored")
        lessons = "".join(f"<li>{escape(x)}</li>" for x in r.get("lessons") or [])
        events_html.append(f"""
  <article class="review">
    <header><h3>{escape(r["event_name"])}</h3><span class="feed-date">{_short_date(r["event_date"])} {r["event_date"][:4]}</span>
      <span class="review-score">{head}</span></header>
    <h4>Lessons</h4><ul class="lessons">{lessons}</ul>
    {_scorecard(r.get("bouts") or [])}
  </article>""")
    if not events_html:
        nxt = f" The first review will be for {escape(first_upcoming)}." if first_upcoming else ""
        events_html.append(f'<p class="feed-none">Results appear here once a predicted event is over and its '
                           f'results are in the data (usually within two days).{nxt}</p>')
    evidence = _evidence_table(board or {})
    min_n = (board or {}).get("min_fights", 30)
    tab = (f'<a class="tab tab-changes" role="tab" id="tab-{RESULTS_ID}" href="#{RESULTS_ID}" '
           f'data-target="{RESULTS_ID}" aria-controls="{RESULTS_ID}">'
           f'<span class="tab-title">Results</span><span class="tab-sub">{escape(sub)}</span></a>')
    option = f'<option value="{RESULTS_ID}">Results ({escape(sub)})</option>'
    panel = f"""
<section class="event-panel results-panel" id="{RESULTS_ID}" data-title="Results" aria-labelledby="h-{RESULTS_ID}">
  <header class="panel-head">
    <h2 id="h-{RESULTS_ID}">Results &amp; lessons learned</h2>
    <div class="meta">{record}</div>
  </header>
  {"".join(events_html)}
  <h3 class="seg-title">Evidence board <span>all tracked fights</span></h3>
  <p class="note">Hit rate compared with what the model itself expected (its average confidence) for each group.
    A group is only flagged once it has at least {min_n} fights and the 95% range of its hit rate excludes the
    expectation; until then it is &ldquo;collecting evidence&rdquo;. Model changes are only made for flagged
    groups, and only if they also pass the walk-forward backtest on past seasons.</p>
  {evidence or '<p class="feed-none">No tracked fights yet.</p>'}
</section>"""
    return tab, option, panel


def _panel_changes(e: dict) -> str:
    changes = sorted(e.get("changes") or [], key=lambda c: c.get("detected_at") or "", reverse=True)
    if not changes:
        return ""
    rows = "".join(f'<li class="feed-row"><div class="feed-when">{escape(_when(c.get("detected_at")))}</div>'
                   f'<ul class="chg-list">{_change_items(c)}</ul></li>' for c in changes)
    return (f'<details class="panel-changes" open><summary>Card changes ({len(changes)})</summary>'
            f'<ul class="feed">{rows}</ul></details>')


def _panel(e: dict, slug: str, is_next: bool) -> str:
    d = date.fromisoformat(e["event_date"])
    facts = [f'<span>Date <b>{d.strftime("%A %d %B %Y")}</b></span>']
    if e.get("location"):
        facts.append(f'<span>Venue <b>{escape(e["location"])}</b></span>')
    facts.append(f"<span>{_bouts(e['bouts'])}</span>" if e["bouts"] else "<span>Card not announced yet</span>")
    if e.get("fetched_at"):
        facts.append(f'<span>Card checked <b>{escape(_when(e["fetched_at"]))}</b></span>')
    if e.get("unmatched"):
        facts.append(f'<span class="warn">{e["unmatched"]} fighter(s) not in UFC data (treated as debuts)</span>')
    if e.get("report"):
        facts.append(f'<a class="standalone" href="{escape(e["report"])}">Standalone page</a>')
    pred = e.get("pred")
    if pred is not None and len(pred):
        body = _bout_grid(pred)
    else:
        body = '<p class="empty">No bouts announced yet, so nothing to predict. Check back after the next schedule fetch.</p>'
    label = '<span class="label-next">Next event</span>' if is_next else ""
    return f"""
<section class="event-panel" id="{slug}" data-title="{escape(e["event_name"])}" aria-labelledby="h-{slug}">
  <header class="panel-head">
    <h2 id="h-{slug}">{escape(e["event_name"])}{label}</h2>
    <div class="meta">{"".join(facts)}</div>
    {_panel_changes(e)}
  </header>
  {body}
</section>"""


def _tab(e: dict, slug: str, is_next: bool, now: datetime | None = None) -> str:
    d = date.fromisoformat(e["event_date"])
    badge = f'{d.strftime("%b")} {d.day}'
    h = e.get("headliner")
    sub = (f'<span class="tab-sub">{escape(h["fighter_1"])} vs {escape(h["fighter_2"])}</span>'
           if h else '<span class="tab-sub">Card TBA</span>')
    now = now or datetime.now(timezone.utc)
    updated = any(_is_recent(c, now) for c in e.get("changes") or [])
    dot = ('<span class="upd-dot" title="Card changed in the last 7 days" '
           'aria-label="card changed recently"></span>') if updated else ""
    return (f'<a class="tab{" next" if is_next else ""}" role="tab" id="tab-{slug}" href="#{slug}" '
            f'data-target="{slug}" aria-controls="{slug}">'
            f'<span class="tab-title">{escape(short_title(e["event_name"]))} '
            f'<span class="date-badge">{badge}</span>{dot}</span>{sub}</a>')


def _option(e: dict, slug: str) -> str:
    d = date.fromisoformat(e["event_date"])
    return f'<option value="{slug}">{escape(short_title(e["event_name"]))} ({d.strftime("%b")} {d.day})</option>'


# Tab switching. Plain ES5, no external code. Without JavaScript every panel is shown
# stacked and the tabs are ordinary in-page links. The last statement marks the page as
# ready, which the tests use to detect script errors in a real browser.
DASHBOARD_JS = """
(function () {
  var root = document.documentElement;
  var panels = Array.prototype.slice.call(document.querySelectorAll(".event-panel"));
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tab[data-target]"));
  var select = document.getElementById("event-select");
  var ids = panels.map(function (p) { return p.id; });
  var baseTitle = document.title;

  function resolve(id) {
    if (ids.indexOf(id) >= 0) { return id; }
    var fallback = root.getAttribute("data-default");
    return ids.indexOf(fallback) >= 0 ? fallback : ids[0];
  }

  function show(id, updateUrl) {
    id = resolve(id);
    panels.forEach(function (p) {
      var on = p.id === id;
      p.hidden = !on;
      if (on) { document.title = p.getAttribute("data-title") + " \\u00b7 " + baseTitle; }
    });
    tabs.forEach(function (t) {
      var on = t.getAttribute("data-target") === id;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.setAttribute("tabindex", on ? "0" : "-1");
      if (on && t.scrollIntoView) { t.scrollIntoView({block: "nearest", inline: "nearest"}); }
    });
    if (select) { select.value = id; }
    if (updateUrl && window.location.hash !== "#" + id) {
      try { history.pushState(null, "", "#" + id); } catch (e) { window.location.hash = id; }
    }
    if (updateUrl && window.goatcounter && window.goatcounter.count) {
      try { window.goatcounter.count({path: "tab/" + id, title: document.title, event: true}); } catch (e) {}
    }
  }

  function currentHash() {
    try { return decodeURIComponent(window.location.hash.slice(1)); } catch (e) { return ""; }
  }

  tabs.forEach(function (t, i) {
    t.addEventListener("click", function (ev) {
      ev.preventDefault();
      show(t.getAttribute("data-target"), true);
    });
    t.addEventListener("keydown", function (ev) {
      var step = ev.key === "ArrowRight" ? 1 : ev.key === "ArrowLeft" ? -1 : 0;
      if (!step) { return; }
      ev.preventDefault();
      var next = tabs[(i + step + tabs.length) % tabs.length];
      show(next.getAttribute("data-target"), true);
      next.focus();
    });
  });
  if (select) {
    select.addEventListener("change", function () { show(select.value, true); });
  }
  window.addEventListener("hashchange", function () { show(currentHash(), false); });
  window.addEventListener("popstate", function () { show(currentHash(), false); });

  root.classList.add("js");
  show(currentHash(), false);
  root.setAttribute("data-dashboard", "ready");
})();
"""


_GOATCOUNTER_CODE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,48}[a-z0-9])?$")


def analytics_tag(goatcounter_code: str | None) -> tuple[str, str]:
    """(<script> for <head>, footer note) for GoatCounter, or ("", "") when no code is set.
    GoatCounter is cookie-free and stores no personal data; it is the one external script the
    dashboard may load (config `analytics.goatcounter_code`)."""
    if not goatcounter_code:
        return "", ""
    if not _GOATCOUNTER_CODE.match(goatcounter_code):
        raise ValueError(f"invalid GoatCounter code {goatcounter_code!r} (lowercase letters, digits, hyphens)")
    tag = (f'<script data-goatcounter="https://{goatcounter_code}.goatcounter.com/count" '
           f'async src="https://gc.zgo.at/count.js"></script>\n')
    note = (" Anonymous visit counts via GoatCounter (no cookies, no personal data): "
            f"https://{goatcounter_code}.goatcounter.com")
    return tag, note


def render_index(entries: list[dict], meta: dict, generated: str | None = None,
                 reviews: list[dict] | None = None, board: dict | None = None,
                 goatcounter_code: str | None = None) -> str:
    """Single-file dashboard of every scheduled card: event tabs (a <select> on narrow
    screens), one `<section class="event-panel" id="<event_slug>">` per event with all bout
    cards, `#<event_slug>` deep links, default = the next event with bouts.

    `entries` come from predict.run_predict_all: event_name, event_date, location, bouts,
    report (standalone html file name or None), unmatched, headliner (main-event prediction or
    None), pred (predict_card output with `_drivers`, or None), fetched_at and changes (line-up
    changes, see upcoming.diff_bouts). `reviews`/`board` are review.load_reviews() output for the
    Results tab. `goatcounter_code` adds the GoatCounter visit counter (off when empty). Self-contained: embedded CSS
    and one inline script, no external resources; everything still shows without JavaScript."""
    generated = generated or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entries = sorted(entries, key=lambda e: (e["event_date"], e["event_name"]))
    slugs, seen = [], set()
    for e in entries:
        s = event_slug(e["event_name"])
        if s in seen:
            s = f"{s}-{e['event_date']}"
        seen.add(s)
        slugs.append(s)
    next_i = next((i for i, e in enumerate(entries) if e.get("headliner")), 0 if entries else None)
    default = slugs[next_i] if next_i is not None else ""

    now = datetime.fromisoformat(generated)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ch_tab, ch_option, ch_panel = _changes_view(entries, slugs)
    first = entries[next_i]["event_name"] if next_i is not None else None
    rs_tab, rs_option, rs_panel = _results_view(reviews, board, first)
    tabs = rs_tab + ch_tab + "".join(_tab(e, s, i == next_i, now) for i, (e, s) in enumerate(zip(entries, slugs)))
    options = "".join(_option(e, s) for e, s in zip(entries, slugs)) + rs_option + ch_option
    panels = ("".join(_panel(e, s, i == next_i) for i, (e, s) in enumerate(zip(entries, slugs)))
              + rs_panel + ch_panel)
    predicted = sum(1 for e in entries if e.get("headliner"))
    analytics, analytics_note = analytics_tag(goatcounter_code)
    n_bouts = sum(len(e["pred"]) for e in entries if e.get("pred") is not None)
    return f"""<!doctype html>
<html lang="en" data-default="{default}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upcoming UFC predictions</title>
<style>{CSS}{DASHBOARD_CSS}</style>
{analytics}</head>
<body>
<header class="appbar">
  <div class="appbar-inner">
    <div class="brand"><span class="kicker">Fight predictions</span><span class="brand-title">Upcoming UFC events</span></div>
    <label class="picker"><span class="sr-only">Choose an event</span>
      <select id="event-select">{options}</select></label>
  </div>
  <nav class="tabs" role="tablist" aria-label="Events">{tabs}</nav>
</header>
<div class="wrap">
  <div class="meta summary">
    <span><b>{len(entries)}</b> scheduled events, <b>{predicted}</b> predicted, <b>{n_bouts}</b> bouts</span>
    <span>Model <b>{escape(str(meta.get("model_version", "")))}</b> ({escape(str(meta.get("model_type", "")))}, data to {escape(str(meta.get("data_cutoff", "")))})</span>
    <span>Generated <b>{escape(generated)}</b></span>
  </div>
  <main>{panels}
  </main>
  <footer>
    Schedule and bouts from the official UFC event pages; cards change often, so later events are
    more likely to change before fight night. {FOOTER_NOTE}{_accuracy_note(meta)}{escape(analytics_note)}
  </footer>
</div>
<script>{DASHBOARD_JS}</script>
</body>
</html>
"""
