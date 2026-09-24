"""Self-contained HTML report for a predicted card (embedded CSS, no scripts, no external
resources, so it works offline and can be emailed as a single file)."""
from __future__ import annotations

from datetime import date, datetime, timezone
from html import escape
from types import SimpleNamespace

import pandas as pd

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

INDEX_CSS = """
.events { display: flex; flex-direction: column; gap: 12px; list-style: none; margin: 0; padding: 0; }
.event { display: grid; grid-template-columns: 78px 1fr minmax(0, 420px); gap: 16px; align-items: center;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.event.next { border-color: var(--pick); }
.when { text-align: center; border-right: 1px solid var(--line); padding-right: 12px; }
.when .dow { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .1em; }
.when .day { font-size: 26px; font-weight: 750; line-height: 1.1; font-variant-numeric: tabular-nums; }
.when .mon { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.info h2 { margin: 0 0 4px; font-size: 17px; line-height: 1.25; }
.info h2 a { text-decoration: none; }
.info h2 a:hover { text-decoration: underline; }
.info .sub { color: var(--muted); font-size: 13px; display: flex; flex-wrap: wrap; gap: 4px 14px; }
.info .sub .warn { color: var(--debut); }
.head { display: flex; flex-direction: column; gap: 6px; }
.head .names { display: flex; justify-content: space-between; gap: 10px; font-size: 14px; font-weight: 600; }
.head .names span { min-width: 0; overflow-wrap: anywhere; }
.head .names .p { font-variant-numeric: tabular-nums; font-weight: 700; }
.head .n1 .p { color: var(--f1); }
.head .n2 { text-align: right; }
.head .n2 .p { color: var(--f2); }
.head .pickline { color: var(--muted); font-size: 12px; }
.head .pickline b { color: var(--pick); font-weight: 650; }
.head.empty { color: var(--muted); font-size: 13px; }
.cta { display: inline-block; margin-top: 8px; font-size: 13px; font-weight: 600; color: var(--pick);
  text-decoration: none; }
.cta:hover { text-decoration: underline; }
.label-next { font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #1b1300;
  background: var(--pick); border-radius: 999px; padding: 1px 7px; vertical-align: 2px; margin-left: 6px; }
@media (max-width: 760px) {
  .event { grid-template-columns: 64px 1fr; }
  .head { grid-column: 1 / -1; }
}
"""


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


def render_html(pred: pd.DataFrame, card, meta: dict, index_link: bool = False) -> str:
    """`pred` is predict.predict_card() output (with `_drivers`); `card` has event_name/event_date.
    `index_link` adds a link to index.html in the same folder (the schedule overview)."""
    d = date.fromisoformat(card.event_date)
    generated = str(pred["predicted_at"].iloc[0]) if len(pred) else ""
    # records, not itertuples: itertuples renames the `_drivers` column
    cards = "".join(_bout_card(SimpleNamespace(**r)) for r in pred.sort_values("bout_order").to_dict("records"))
    try:
        acc = meta["evaluation"]["metrics"][meta["model_type"]]["test"]["accuracy"]
        acc_note = f" On held-out test fights this model picked the winner {acc:.0%} of the time."
    except (KeyError, TypeError):
        acc_note = ""
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
    {'<a class="back" href="index.html">&larr; All upcoming events</a>' if index_link else ""}
    <div class="kicker">Fight predictions</div>
    <h1>{escape(card.event_name)}</h1>
    <div class="meta">
      <span>Date <b>{d.strftime("%A %d %B %Y")}</b></span>{f'''
      <span>Venue <b>{escape(card.location)}</b></span>''' if getattr(card, "location", None) else ""}
      <span>Model <b>{escape(str(meta.get("model_version", "")))}</b> ({escape(str(meta.get("model_type", "")))}, data to {escape(str(meta.get("data_cutoff", "")))})</span>
      <span>Generated <b>{escape(generated)}</b></span>
      <span>{len(pred)} bouts</span>
    </div>
  </header>
  <main class="grid">{cards}
  </main>
  <footer>
    Probabilities average both fighter orders, so they always sum to 100%. Key factors are the
    model's feature contributions (in log-odds, combined over both orders): each tag shows the
    values for the fighter on the left vs the right, and the bar shows its size relative to the
    strongest factor in that bout. Factors explain this model's estimate, not what will happen
    in the fight.{acc_note}
  </footer>
</div>
</body>
</html>
"""


def _index_row(e: dict, is_next: bool) -> str:
    d = date.fromisoformat(e["event_date"])
    name = escape(e["event_name"])
    title = f'<a href="{escape(e["report"])}">{name}</a>' if e.get("report") else name
    if is_next:
        title += '<span class="label-next">Next</span>'
    sub = []
    if e.get("location"):
        sub.append(f"<span>{escape(e['location'])}</span>")
    sub.append(f"<span>{e['bouts']} bouts</span>" if e["bouts"] else "<span>Card not announced yet</span>")
    if e.get("unmatched"):
        sub.append(f'<span class="warn">{e["unmatched"]} fighter(s) not in UFC data (debuts?)</span>')

    h = e.get("headliner")
    if h:
        p1, p2 = float(h["p_fighter_1"]), float(h["p_fighter_2"])
        aria = f"{h['fighter_1']} {_pct(p1)}, {h['fighter_2']} {_pct(p2)}"
        wc = f" · {escape(str(h['weight_class']))}" if isinstance(h.get("weight_class"), str) else ""
        head = f"""<div class="head">
      <div class="names"><span class="n1">{escape(h["fighter_1"])} <span class="p">{_pct(p1)}</span></span><span class="n2"><span class="p">{_pct(p2)}</span> {escape(h["fighter_2"])}</span></div>
      <div class="bar" role="img" aria-label="{escape(aria)}"><span class="seg f1" style="width:{100 * p1:.1f}%"></span><span class="seg f2" style="width:{100 * p2:.1f}%"></span></div>
      <div class="pickline">Main event{wc} · pick <b>{escape(h["predicted_winner"])}</b> ({float(h["confidence"]):.0%})</div>
    </div>"""
        cta = f'<a class="cta" href="{escape(e["report"])}">Full card predictions &rarr;</a>'
    else:
        head = '<div class="head empty">No bouts announced yet, so nothing to predict.</div>'
        cta = ""
    return f"""
  <li class="event{' next' if is_next else ''}">
    <div class="when"><div class="dow">{d.strftime("%a")}</div><div class="day">{d.day}</div><div class="mon">{d.strftime("%b %Y")}</div></div>
    <div class="info"><h2>{title}</h2><div class="sub">{"".join(sub)}</div>{cta}</div>
    {head}
  </li>"""


def render_index(entries: list[dict], meta: dict, generated: str | None = None) -> str:
    """Schedule overview: one row per scheduled event (chronological), linking to each report.

    `entries` come from predict.run_predict_all: event_name, event_date, location, bouts,
    report (html file name or None), unmatched, headliner (main-event prediction or None)."""
    generated = generated or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    entries = sorted(entries, key=lambda e: (e["event_date"], e["event_name"]))
    next_i = next((i for i, e in enumerate(entries) if e.get("headliner")), None)
    rows = "".join(_index_row(e, i == next_i) for i, e in enumerate(entries))
    predicted = sum(1 for e in entries if e.get("headliner"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upcoming UFC predictions</title>
<style>{CSS}{INDEX_CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="top">
    <div class="kicker">Fight predictions</div>
    <h1>Upcoming UFC events</h1>
    <div class="meta">
      <span><b>{len(entries)}</b> scheduled events, <b>{predicted}</b> with predictions</span>
      <span>Model <b>{escape(str(meta.get("model_version", "")))}</b> ({escape(str(meta.get("model_type", "")))}, data to {escape(str(meta.get("data_cutoff", "")))})</span>
      <span>Generated <b>{escape(generated)}</b></span>
    </div>
  </header>
  <ul class="events">{rows}
  </ul>
  <footer>
    Schedule and bouts from the official UFC event pages; cards change often, so later events are
    more likely to change before fight night. Each bar shows the main event's win probability
    (averaged over both fighter orders). Open an event for every bout and its key factors.
  </footer>
</div>
</body>
</html>
"""
