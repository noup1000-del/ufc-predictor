"""Self-contained HTML report for a predicted card (embedded CSS, no scripts, no external
resources, so it works offline and can be emailed as a single file)."""
from __future__ import annotations

from datetime import date
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


def render_html(pred: pd.DataFrame, card, meta: dict) -> str:
    """`pred` is predict.predict_card() output (with `_drivers`); `card` has event_name/event_date."""
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
    <div class="kicker">Fight predictions</div>
    <h1>{escape(card.event_name)}</h1>
    <div class="meta">
      <span>Date <b>{d.strftime("%A %d %B %Y")}</b></span>
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
