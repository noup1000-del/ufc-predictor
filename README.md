# 🥊 UFC Matchup Intelligence & Predictive Engine

An end-to-end, leak-free machine learning pipeline and interactive dashboard predicting upcoming UFC bouts. Built on calibrated probabilistic models (LightGBM / Logistic Regression), fold-strict imputation, and multi-variable tree feature explainability.

Live Interactive Dashboard: **[https://noup1000-del.github.io/ufc-predictor/](https://noup1000-del.github.io/ufc-predictor/)**

---

## ⚡ Key Highlights

- **Leak-Free Architecture:** Strict temporal ordering across all training, validation, and test splits. Feature provenance assertions verify that all historical features satisfy `max(source_event_date) < fight_date`.
- **Fold-Strict Imputation:** Missing physical attributes (height-to-reach linear regression and division medians) are fitted strictly on preceding training slices, preventing future data leakage into evaluation sets.
- **Probabilistic Calibration:** Evaluated on Expected Calibration Error (ECE), Brier Score decomposition (Reliability vs. Resolution), and 95% Wilson binomial score intervals.
- **Explainable Predictions:** Individual fight probability bars are backed by tree feature contributions (`pred_contrib=True` / standardized logistic coefficients), highlighting the top tactical drivers (age differentials, strike absorption rates, takedown defense, control time).
- **Automated Scheduling & Scraping:** Scrapes confirmed bout rosters directly from [ufc.com/events](https://www.ufc.com/events) with per-host polite crawl delays and name aliasing.
- **Zero-Dependency Dashboards:** Generates self-contained, offline-compatible, dark-themed HTML reports and a multi-event tabbed dashboard deployed automatically via GitHub Pages.

---

## 📊 Empirical Walk-Forward Backtesting

Every training run passes a promotion gate that includes a walk-forward backtest over rolling historical origins (also available on its own as `python run_pipeline.py --backtest`). For each slice the model is tuned on the year before the cutoff, refit on everything before it (imputer included), and tested on the following period, without lookahead:

| Cutoff Date | Test Period | Model | N | ROC-AUC | Brier Score | Log Loss | ECE | Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2023-01-01** | 2023-01-14 – 2023-12-16 | LightGBM | 504 | 0.636 | 0.2365 | 0.6657 | 0.035 | 58.9% |
| **2024-01-01** | 2024-01-13 – 2024-12-14 | LightGBM | 513 | 0.676 | 0.2277 | 0.6468 | 0.044 | 61.6% |
| **2025-01-01** | 2025-01-11 – 2026-09-19 | LightGBM | 912 | 0.683 | 0.2249 | 0.6410 | 0.029 | **64.3%** |

*(Baseline "more UFC wins": AUC 0.47–0.51, Log Loss ~0.70 across all test windows.)* Logistic regression lands within ±0.01 AUC of LightGBM on every slice; the production model is whichever has the lower validation log loss. N counts fights; metrics use the symmetric probability of both fighter orders. Full per-slice results, including calibration tables, are written to `models/backtest_report.json`.

---

## 🛠️ Project Architecture

```
 Greco1899 archive (ufcstats CSVs, pinned by commit)          ufc.com/events  (ESPN / manual JSON fallback)
                │                                                          │
            ingest ── name → fighter_id, once ──┐                upcoming cards → data/raw/cards/*.json
                │                               │                          │
             clean ── typed, validated tables   │                          │
                │                               │                          │
           features ── pre-fight state only, provenance-checked ◄──────────┤  same code path
                │                                                          │
             train ── baseline · logistic · LightGBM, backtest, promotion gate
                │                                                          │
             track ◄── results vs. saved predictions                  predict ── CSV + HTML per card
                                                                            │
                                                            outputs/predictions/index.html ──► GitHub Pages
```

| Stage | Module | What it does |
| :--- | :--- | :--- |
| Ingest | `src/ingest.py`, `src/matching.py` | Downloads the [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) CSVs for a pinned commit, maps fighter names to ufcstats IDs once (overrides, exact match, age/weight disambiguation; never fuzzy), merges incrementally and idempotently. Unmappable fights go to `data/raw/ingest_dropped.csv`. |
| Clean | `src/clean.py` | Typed tables with a declared schema, a configurable date cutoff (2005), stat validation and a JSON validation report. |
| Features | `src/features.py` | Cumulative per-fighter state read with `merge_asof(allow_exact_matches=False)`: record, streaks, layoff, finish/KO/sub rates, per-15-minute striking and grappling rates, last-3 windows, physical and context features. Debuts stay `NaN` with `is_debut = 1`, not zeros. Each fight appears in both orientations. |
| Train | `src/train.py` | Time-based splits only, three models side by side, fold isolation (`FoldLeakError`), missingness report, leakage flag for dominant features, promotion gate (provenance + backtest + full test suite) before `models/latest.pkl` changes. |
| Predict | `src/predict.py`, `src/upcoming.py`, `src/report.py` | Next card or every scheduled card, symmetric probabilities `p = (p(A,B) + 1 − p(B,A)) / 2`, top-3 drivers per fighter, CSV + standalone HTML per card, tabbed dashboard. Unmatched names are shown as debuts, never guessed. |
| Track | `src/track.py` | Joins finished fights to saved predictions by fighter ID and date and appends to `outputs/tracking/results_log.csv` with running accuracy and Brier score. |

`CLAUDE.md` is the full specification (schemas, leakage rules, data-source rules) and is kept in sync with the code.

---

## 🚀 Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/noup1000-del/ufc-predictor.git
cd ufc-predictor
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_pipeline.py --full     # first run: download archive, clean, features, train, predict
python run_pipeline.py --predict-all
```

Open `outputs/predictions/index.html` in a browser. `data/`, `models/` and `logs/` are created locally and are not committed.

## 🧭 Commands

```bash
python run_pipeline.py --update          # default: ingest new → clean → features → track → train (if needed) → predict
python run_pipeline.py --full            # rebuild everything from the cached source, always retrain
python run_pipeline.py --predict-only    # predict the next card with models/latest.pkl
python run_pipeline.py --predict-all     # fetch the ufc.com schedule, predict every card, write the dashboard
python run_pipeline.py --backtest        # walk-forward backtest → models/backtest_report.json
python run_pipeline.py --stage features  # one stage: ingest|clean|features|track|train|predict|backtest
python run_pipeline.py --dry-run         # show the plan without running anything
```

Options: `--card PATH` (predict a specific card JSON), `--force-train`, `--no-predict` (stop after train, for the post-event job). Every stage logs row counts, output also goes to `logs/pipeline.log`, and any failure exits with code 1 so a scheduler can detect it.

**Overriding a card:** put a hand-made card in `data/raw/upcoming_card.json` (format: `tests/fixtures/upcoming_card.example.json`); it takes precedence over ufc.com. **Name mismatches:** if the log reports a fighter without an ID who does have UFC fights (ufc.com and ufcstats sometimes spell names differently), add a verified line to `upcoming_aliases.csv`.

**Scheduling:** a Monday results/tracking job and a Wednesday prediction job via Windows Task Scheduler (`scripts/run_pipeline.bat`), cron or GitHub Actions; see [`docs/scheduling.md`](docs/scheduling.md). The Pages workflow (`.github/workflows/pages.yml`) republishes the dashboard whenever files under `outputs/predictions/` change on `main`.

---

## 🧪 Testing

```bash
python -m pytest
```

The suite covers parsers, ingest (outcome mapping, round aggregation, renamed-event duplicates, same-name disambiguation, overrides, idempotency), name matching, the leakage test (recomputing features from strictly earlier fights only), prediction symmetry, fold isolation, the promotion gate, tracking, the ufc.com parsers, and the dashboard. The dashboard test runs the page's script in headless Edge/Chrome and is skipped where neither is installed. Tests use saved fixtures, never live requests.

---

## 📚 Data Sources & Responsible Use

- **Historical fights:** CSV exports from [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats) (GPL-3.0), which mirrors [ufcstats.com](http://ufcstats.com). Only the CSVs are downloaded; none of that repository's code is used and no data is committed here. ufcstats.com itself is not scraped.
- **Upcoming cards:** the official event pages at [ufc.com/events](https://www.ufc.com/events), fetched with an identifying User-Agent, at most one request per second overall and the 15-second crawl delay from ufc.com's `robots.txt`. Only names, weight classes, dates and venues are read. ESPN's public scoreboard API is the fallback.
- **Never:** bypassing bot protection or access blocks, betting-odds sources, or features based on nationality or ethnicity.

---

## ⚠️ Limitations

- A realistic ceiling for fight prediction is around 60–65% accuracy; the model is built to be well calibrated, not to beat that ceiling.
- Debutants have no UFC history, so their predictions rest mostly on physical attributes and context.
- Cards change up to fight night. Predictions for later events use the card as announced when they were generated.
- Explanations describe what drove *this model's* estimate. Tree models are not monotonic, so a driver can favour the fighter with the "worse-looking" number.

This project is for data-science research and entertainment. Predictions are not betting advice.
