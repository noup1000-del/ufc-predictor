# UFC Fight Predictor — Project Spec

This file is the source of truth for this project. Read it fully before writing code, and follow it in every session. If something here conflicts with a request, ask before deviating.

## Goal

A Python pipeline that:
1. Imports UFC fight history from a pre-scraped public ufcstats.com archive (see "Data sources") and stores it locally.
2. Updates incrementally: each run only adds events/fights/fighters not already saved.
3. Builds features using only information available **before** each fight.
4. Trains a model to predict fight winners.
5. Fetches the next upcoming card and outputs win probabilities.
6. Logs past predictions against actual results so accuracy can be tracked over time.

Realistic target: ~60–65% accuracy on held-out recent fights, with well-calibrated probabilities. Honest evaluation matters more than a high number.

## Tech stack

- Python 3.11+, virtual environment in `.venv/`
- `requests`, `beautifulsoup4`, `lxml`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyyaml`, `pytest`
- Keep `requirements.txt` up to date with pinned versions.
- Use the `logging` module, not bare `print`, in `src/`.

## Folder structure

```
ufc-predictor/
├── CLAUDE.md
├── config.yaml             # settings: paths, rate limit, date cutoffs, model params
├── requirements.txt
├── ingest_overrides.csv    # committed manual fighter-ID fixes for name collisions (see ingest rules)
├── run_pipeline.py         # entry point, runs stages in order
├── data/
│   ├── raw/
│   │   ├── source/greco/<commit_sha>/   # downloaded Greco1899 CSVs, pinned by commit
│   │   ├── source/manifest.json         # commit SHA + row counts of the last ingest
│   │   ├── espn/           # cached ESPN scoreboard JSON
│   │   ├── upcoming_card.json   # optional manual card override
│   │   ├── ingest_dropped.csv   # fights the last ingest could not map, with reason
│   │   ├── events.csv
│   │   ├── fights.csv
│   │   ├── fight_stats.csv
│   │   └── fighters.csv
│   ├── processed/          # cleaned and typed tables
│   └── features/           # one row per fight, model-ready
├── models/                 # trained model + metadata (date trained, metrics, feature list)
├── outputs/
│   ├── predictions/        # <event_date>_<event_slug>.csv
│   └── tracking/
│       └── results_log.csv # every prediction + actual outcome once known
├── src/
│   ├── __init__.py
│   ├── config.py           # loads config.yaml, resolves project paths
│   ├── http.py             # shared session, rate limiting, retries, download cache
│   ├── ingest.py           # Greco1899 archive -> raw schemas (the only place names are joined)
│   ├── matching.py         # name normalisation + fighter-name -> fighter_id matching
│   ├── upcoming.py         # upcoming card: manual JSON override, else ESPN API
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── track.py
├── tests/
└── notebooks/              # exploration only; pipeline must not depend on these
```

`data/`, `models/` and `.venv/` go in `.gitignore`, except `outputs/` which should be committed.

## Data sources

**ufcstats.com is not scraped.** Since 2026-09 it serves a JavaScript proof-of-work browser challenge to automated clients. Do not attempt to solve or bypass it. `src/http.py` detects challenge pages, raises `FetchError`, and never caches them.

**Historical data: Greco1899/scrape_ufc_stats** (https://github.com/Greco1899/scrape_ufc_stats, GPL-3.0, refreshed automatically several times a week). We download only its CSVs, never its code:

| File | Columns | Used for |
|---|---|---|
| `ufc_event_details.csv` | EVENT, URL, DATE, LOCATION | events |
| `ufc_fight_results.csv` | EVENT, BOUT, OUTCOME, WEIGHTCLASS, METHOD, ROUND, TIME, TIME FORMAT, REFEREE, DETAILS, URL | fights |
| `ufc_fight_stats.csv` | EVENT, BOUT, ROUND, FIGHTER, KD, SIG.STR., …, HEAD … GROUND (one row per fighter per round) | fight_stats (summed over rounds) |
| `ufc_fighter_details.csv` | FIRST, LAST, NICKNAME, URL | fighters, name index |
| `ufc_fighter_tott.csv` | FIGHTER, HEIGHT, WEIGHT, REACH, STANCE, DOB, URL | fighters (bio), name index |
| `ufc_fight_details.csv` | EVENT, BOUT, URL | not needed (same info as results) |

IDs are the ufcstats hex IDs taken from the URL columns (`.../event-details/<event_id>`, `.../fight-details/<fight_id>`, `.../fighter-details/<fighter_id>`). The archive has no nationality field; do not infer nationality.

Known quirks (handled in `src/ingest.py`):
- Fight results and fight stats identify fighters **by name only** (no fighter URL); fight stats also has no fight URL (only EVENT + BOUT names).
- Some fights appear twice under an old and a new event name (e.g. "UFC Fight Night: X" renamed "Noche UFC: X"); keep the row whose event name exists in the events file.
- `OUTCOME` is `W/L`, `L/W`, `D/D` or `NC/NC`, aligned with the order of names in `BOUT` ("A vs. B").
- Some fighters' names differ between the two fighter files (renames); index both.
- Exact duplicate stat rows exist; stat rows with an empty ROUND mean "no stats recorded".

**Upcoming cards:** `src/upcoming.py` uses, in order:
1. `data/raw/upcoming_card.json` if present and its date is today or later (manual override; format in `tests/fixtures/upcoming_card.example.json`; may set `fighter_1_id`/`fighter_2_id` directly).
2. ESPN's public scoreboard API: `https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard?dates=<YYYYMMDD>-<YYYYMMDD>`. Non-UFC-card events (e.g. Dana White's Contender Series) are skipped. If ESPN refuses the request (e.g. 403), fail clearly and ask for the manual file; do not work around blocks.

## Download & ingest rules

- **Rate limit:** max 1 request per second, enforced in `src/http.py` for every request (GitHub, ESPN, anything).
- Send a descriptive `User-Agent` (`UFCFightPredictor/0.1 (DataScienceResearch; non-commercial)`). Use timeouts (15s) and retry with exponential backoff (3 attempts).
- **Pin and cache the source:** each run asks the GitHub API for the latest commit SHA of Greco1899 and downloads the CSVs for that SHA to `data/raw/source/greco/<sha>/` only if not already there. `--full` rebuilds from the cached source without re-downloading. Record SHA and row counts in `data/raw/source/manifest.json`.
- **Incremental:** merge only rows whose keys are not already in the raw tables (events by `event_id`, fights by `fight_id`, stats by `(fight_id, fighter_id)`, fighters by `fighter_id`). Fighters, fights and stats are written before events.
- **Errors:** handle each fight individually; log the ID and reason and continue. Fights that cannot be mapped are written to `data/raw/ingest_dropped.csv` with a reason, and a count is logged at the end.
- Writes must be idempotent: running the ingest twice in a row adds nothing the second time and never creates duplicate rows.

**Name-matching exception (strictly limited to `src/ingest.py` and to upcoming-card matching in `src/upcoming.py`/`src/matching.py`).** Because the archive gives fighters by name only, ingest maps names to `fighter_id` once, at import time:
1. `ingest_overrides.csv` (`fight_id, name, fighter_id, note`) wins if it has an entry.
2. Exact match on the normalised name (accents stripped, lowercase, punctuation removed) against both fighter files. Must be unique.
3. If several fighters share the name: drop candidates whose age at the fight would be < 18 or > 50, then pick the candidate whose listed weight is clearly closest (≥ 15 lb closer than the next) to the bout's weight-class limit.
4. Otherwise the fight is dropped and listed in `ingest_dropped.csv` for a manual override. Never fuzzy-match historical names.
Every stage after ingest joins on IDs only.

For upcoming cards (ESPN names), `src/matching.py` may additionally use a fuzzy fallback: accept only if similarity ≥ 0.92 and ≥ 0.05 ahead of the runner-up. Unmatched names are reported in the prediction output, never silently guessed.

## Data schemas

**events.csv:** `event_id, event_name, event_date (YYYY-MM-DD), location, scraped_at` (`scraped_at` = when our ingest added the row)

**fights.csv:** `fight_id, event_id, event_date, bout_order, fighter_1_id, fighter_2_id, fighter_1_name, fighter_2_name, winner_id (empty for draw/NC), result (win/draw/nc), method, end_round, end_time_sec, scheduled_rounds, weight_class, is_title_fight`

**fight_stats.csv** (two rows per fight, one per fighter, fight totals):
`fight_id, fighter_id, opponent_id, kd, sig_str_landed, sig_str_attempted, total_str_landed, total_str_attempted, td_landed, td_attempted, sub_att, reversals, ctrl_sec, sig_head_landed, sig_body_landed, sig_leg_landed, sig_distance_landed, sig_clinch_landed, sig_ground_landed, fight_duration_sec`

**fighters.csv:** `fighter_id, name, nickname, height_in, weight_lb, reach_in, stance, dob (YYYY-MM-DD), scraped_at`

Rules:
- **Always join on IDs, never on names.** Several fighters share names. (Sole exception: the name→ID mapping inside ingest/upcoming-card matching described above.)
- `bout_order`: 1 = the first bout listed for the event in the source (the main event).
- `end_time_sec` is the clock time in the final round; `fight_duration_sec` is total fight time (from the round lengths in the source's time format).
- `weight_class` is the division name without "UFC", "Title", "Interim", "Bout" (e.g. `Women's Strawweight`); NaN when the source names no division (early tournaments). `is_title_fight` is 1 for "Title Bout" rows that are not tournament bouts.
- Parse `x of y` strings into landed/attempted integers, `m:ss` into seconds, heights like `5' 11"` into inches, reach like `72"` into inches. Unknown → `NaN`, not a guessed default.

## Stage 2: clean.py

- Read `data/raw/`, write typed, validated tables to `data/processed/`.
- Drop fights before 2005-01-01 (sparse stats) — make the cutoff configurable.
- Keep draws/NC in the history (they count as fights for experience) but exclude them as training targets.
- Validate: no duplicate IDs, dates parse, every fight has exactly two stat rows (log exceptions).
- Added columns on `fights`: `is_target` (True only for `result == win`), `method_group` (`ko_tko`, `submission`, `decision`, `dq`, `other`), `has_stats` (False if the fight has no valid stat pair; its stat rows are removed).
- Fights before `min_date` go to `data/processed/fights_prior.csv` (same columns, `is_target = False`, `has_stats = False`, no stat rows). Features use them **only** for record/experience features (fights, wins, losses, streaks, KO/sub losses, layoff, weight-class history), never as targets and never in striking/grappling rates.
- Dtypes are declared once in `clean.SCHEMA`; every later stage reads processed tables via `clean.load_processed()`.
- Error-level validation failures exit non-zero; the full report is written to `data/processed/clean_report.json`.

## Stage 3: features.py — the most important rules

**No leakage.** Features for a fight on date D may only use fights with `event_date < D` (strictly earlier; fights on the same card are excluded). Never use career totals from a fighter's current profile page. Age is computed at the fight date.

Per-fighter pre-fight features:
- Experience: UFC fights, wins, losses, win rate (smoothed, e.g. (wins+1)/(fights+2)), current win/loss streak, days since last fight, is_debut flag
- Outcomes: finish rate for wins, KO/TKO losses, submission losses
- Striking (per 15 min of fight time, over prior fights): sig. strikes landed, sig. strikes absorbed, sig. strike accuracy, sig. strike defense, knockdowns
- Grappling: takedowns landed per 15 min, takedown accuracy, takedown defense, submission attempts per 15 min, control seconds per 15 min
- Recency: the same striking/grappling stats over the last 3 fights (or exponentially weighted)
- Physical: height, reach, age at fight, stance
- Context: weight class, whether it's the fighter's first fight in this weight class

For debut fighters, rolling stats are `NaN` plus `is_debut = 1`. Do not fill with zeros (a debut is not a fighter who lands zero strikes). LightGBM handles NaN; for logistic regression use median imputation fitted on training data only.

Matchup features: difference (fighter_1 − fighter_2) for numeric features, and a stance matchup category (orthodox vs southpaw, etc.).

**Symmetry:** the training set must contain each fight in both orientations (A vs B with target 1, B vs A with target 0), with both copies always in the same train/test split. At prediction time, predict both orientations and average: `p = (p(A,B) + 1 − p(B,A)) / 2`.

Output: `data/features/fight_features.csv`, one row per fight orientation, with `fight_id, event_date, fighter_1_id, fighter_2_id, target` plus features.

Implementation notes (src/features.py):
- History is collapsed to one cumulative state per (fighter, event_date); a fight on date D reads the latest state with `hist_date < D` via `merge_asof(allow_exact_matches=False)`. A runtime assertion rejects any state dated on/after D. `matchup_features()` is the single code path for training and prediction.
- Streak: +n consecutive wins / −n consecutive losses; a draw resets to 0; an NC leaves it unchanged. Within one date (old tournaments) bouts are ordered by `bout_order` descending.
- Rates (per 15 min, accuracy, defense) use only post-cutoff fights with stats; each fight contributes to a rate only if its numerator and denominator exist. Defense = opponent misses / opponent attempts; NaN when the opponent never attempted (undefined, not 0). Last-3 windows run over fights with stats.
- Only `is_target` fights become rows. `orientation` 0 = source order, 1 = swapped. `split` is assigned from `event_date` (config `split`), so both orientations always share a split. `data/features/feature_list.json` lists numeric/categorical columns.
- `features.symmetric_probability(p_ab, p_ba)` implements the averaging rule; train and predict must use it.

## Stage 4: train.py

- **Time-based split only, never random.** Example (configurable): train on fights before 2024-01-01, validate on 2024, test on 2025 onward. For the production model, retrain on all data up to the latest event after evaluation.
- Models, in this order:
  1. Baseline: pick the fighter with more UFC wins (and a coin flip for ties).
  2. Logistic regression on standardized difference features.
  3. LightGBM, tuned lightly on the validation set.
- Metrics: log loss, Brier score, accuracy, and a calibration table (bins of predicted probability vs actual win rate). Report all models side by side.
- Save the model to `models/model_<YYYY-MM-DD>.pkl` and a JSON with training date, data cutoff, feature list and metrics. Keep `models/latest.pkl` pointing to the current one.
- Print feature importances. If any single feature is suspiciously dominant, flag it as possible leakage.
- **Bio missingness leaks the future:** ufcstats fills in reach/height/DOB over time for fighters who stay, so historically a missing value marks short careers (train: fighters with missing reach won 9%; recent debutants with missing reach win ~50%). `train.PhysicalImputer` (fitted on the training rows, stored in the model) fills height/reach/age before both models; the feature file keeps NaN. `train.missingness_report` logs target rate by missingness every run; investigate any new feature whose missingness is far from 50%.
- Metrics are per fight on symmetric probabilities; ties at p = 0.5 use a deterministic coin flip. The production model is the logistic/LightGBM model with the lower validation log loss, refit on all data. `models/latest.pkl` / `latest.json` are copies (no symlinks on Windows).

## Stage 5: predict.py

- Get the next card from `src/upcoming.py` (manual `data/raw/upcoming_card.json` first, else ESPN).
- Map fighter names to `fighter_id` with `src/matching.py`. Fighters not in `fighters.csv` are treated as debuts (`is_debut = 1`, physical stats NaN) and listed in the console output; unmatched names must be visible, never guessed.
- Build features as of the event date using the same code path as training (reuse functions from `features.py`; no duplicated logic).
- Output `outputs/predictions/<event_date>_<event_slug>.csv`: `event_date, weight_class, fighter_1, fighter_2, p_fighter_1, p_fighter_2, predicted_winner, confidence, model_version, predicted_at`.
- Also print a readable table to the console.

## Stage 6: track.py

- After an event completes, match its results to the saved predictions and append to `outputs/tracking/results_log.csv`: prediction, actual winner, correct (bool), probability.
- Print running accuracy and Brier score overall and for the last 5 events.

## run_pipeline.py

```
python run_pipeline.py --update     # default: ingest new → clean → features → track → train → predict
python run_pipeline.py --full       # rebuild everything from the cached source (no re-downloading)
python run_pipeline.py --predict-only
python run_pipeline.py --stage features   # run a single stage
```

Log start/end and row counts for each stage. Exit with a non-zero code on failure so a scheduler can detect it.

## Tests (tests/)

- Parsers: `x of y`, `m:ss`, height, reach, and `--` handling.
- Leakage test: for a sample of fights, recompute features from only earlier fights and assert they match; assert no feature row uses a fight dated on or after its own fight date.
- Symmetry test: `p(A beats B) + p(B beats A) ≈ 1` after averaging.
- Idempotency test: running the ingest's merge step twice doesn't add rows.
- Ingest tests: outcome → winner mapping, round aggregation, renamed-event duplicates, same-name disambiguation, overrides.
- Matching tests: normalisation, fuzzy thresholds, ambiguity handling.
- Use saved fixtures in `tests/fixtures/` (excerpts of the Greco1899 CSVs, ESPN JSON), not live requests.

## Build order

Build one milestone at a time. Stop after each one, show results, and wait for confirmation before continuing.

1. **Setup:** folders, `config.yaml`, `requirements.txt`, `.gitignore`, `src/http.py` with rate limit + cache.
2. **Import (replaces the former Scraper + Full backfill milestones):** `src/ingest.py` maps the full Greco1899 archive into the raw schemas; `src/upcoming.py` + `src/matching.py` for upcoming cards. Run validation (unique keys, 2 stat rows per fight, landed ≤ attempted, strike-sum consistency, control time ≤ duration) and report shapes, nulls, dropped fights and spot checks.
3. **Clean:** processed tables + validation report.
4. **Features:** with leakage and symmetry tests passing.
5. **Train:** baseline vs logistic regression vs LightGBM report.
6. **Predict:** predictions for the next upcoming card.
7. **Track + run_pipeline.py:** end-to-end `--update` run.
8. **Scheduling:** instructions for a weekly run (Windows Task Scheduler, cron, or GitHub Actions).

Note for Features: non-UFC records (PFL, ONE, KSW, Cage Warriors, …) are planned as extra experience features; `fights.csv`/`fight_stats.csv` stay UFC-only. That needs a data source decision first — ask before adding one.

## Don'ts

- Don't fill unknown values with made-up defaults (e.g. height = 70).
- Don't use bare `except:` or swallow errors silently.
- Don't use random train/test splits.
- Don't fetch data from other sites (e.g. betting odds) unless asked. Approved sources: the Greco1899 GitHub repo (and GitHub API for its commit SHA) and the ESPN scoreboard API.
- Don't bypass bot protection, CAPTCHAs or access blocks on any site.
- Don't create features based on nationality or ethnicity.
- Don't duplicate feature logic between training and prediction.



## Potential Enhancements for Later Iterations
- Elo / Glicko Rating System: Raw win/loss counts treat a win over a top-5 contender the same as a win over an unranked fighter. An Elo or TrueSkill feature built chronologically adds predictive signal.

- Days of Layoff & Age Curves: Non-linear effects (e.g., fighters over 35 in flyweight/bantamweight drop off sharply compared to heavyweights). Adding an interaction between weight class and age can yield gains.

- Betting Market Baseline: Once the initial pipeline is working, tracking against closing market odds (Pinnacle/Bet365) provides a realistic benchmark to determine whether the model captures market inefficiencies.