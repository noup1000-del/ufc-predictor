# Scheduling the weekly run

## What runs when

UFC events are almost always on Saturday. The historical source (Greco1899) usually adds a
completed event within 1–3 days (commits land around 18:00 UTC). So there are two weekly jobs:

| Job | When | Command | What it does |
|---|---|---|---|
| Results & tracking | **Monday** morning | `run_pipeline.py --update --no-predict` | Imports Saturday's results, rebuilds clean and features, logs last week's predictions against the results, retrains (new data → new model) |
| Predictions | **Wednesday** evening | `run_pipeline.py --update` | Same as above (picks up results the source added late, no-op otherwise), then predicts the next card (fetched from ufc.com) |
| Schedule overview (optional) | after the Wednesday job | `run_pipeline.py --predict-all` | Re-fetches the ufc.com schedule and predicts every announced card, plus `outputs/predictions/index.html` (~2 min: ufc.com asks for 15 s between requests) |

The next card comes from ufc.com/events and is synced to `data/raw/upcoming_card.json`, so
there is no weekly manual step. To override it, write your own `upcoming_card.json`
(format: `tests/fixtures/upcoming_card.example.json`); a hand-made file wins over ufc.com.
If ufc.com refuses (403 / challenge page), the last synced card is used, then ESPN; if
nothing works the job fails with exit code 1 and asks for the manual file. When the log
says a fighter has no fighter_id but you know they have UFC fights (ufc.com spells some
names differently from ufcstats), add a verified line to `upcoming_aliases.csv`.

Every run appends to `logs/pipeline.log`. The exit code is 0 on success and 1 if any stage
failed (the pipeline stops at that stage). A failed Monday run is harmless: the next run
catches up, because ingest and tracking are incremental and idempotent.

Commit `outputs/` after the Wednesday run (the predictions) and after the Monday run
(`outputs/tracking/results_log.csv`), so the record of predictions vs results is kept.

## Windows Task Scheduler

`scripts/run_pipeline.bat` changes to the project folder, runs `.venv\Scripts\python.exe
run_pipeline.py` with the given arguments, and returns its exit code.

Create both tasks from a terminal (adjust the path and times):

```bat
schtasks /Create /TN "UFC Predictor\Monday results" /SC WEEKLY /D MON /ST 09:00 ^
  /TR "\"C:\Users\Gebruiker\ufc-predictor\scripts\run_pipeline.bat\" --update --no-predict"

schtasks /Create /TN "UFC Predictor\Wednesday predict" /SC WEEKLY /D WED /ST 19:00 ^
  /TR "\"C:\Users\Gebruiker\ufc-predictor\scripts\run_pipeline.bat\" --update"
```

Then, in Task Scheduler (`taskschd.msc` → *UFC Predictor* folder) for each task:
- *General*: "Run whether user is logged on or not" (no console window pops up; needs your
  password once). A console window is harmless if you prefer "only when logged on".
- *Settings*: tick "Run task as soon as possible after a scheduled start is missed" (laptop
  asleep on Monday morning), and "Stop the task if it runs longer than 1 hour".
- *Conditions*: untick "Start only if on AC power" on a laptop if needed.

Check results: *Last Run Result* shows `0x0` on success and `0x1` on failure; details are
in `logs\pipeline.log`. Test a task immediately with `schtasks /Run /TN "UFC Predictor\Wednesday predict"`.

`pythonw.exe` is not needed: with "Run whether user is logged on or not" no window is
shown, and the batch file keeps the exit code, which Task Scheduler needs to report failures.
To call Python directly instead of the batch file, set *Program* to
`C:\Users\Gebruiker\ufc-predictor\.venv\Scripts\python.exe`, *Arguments* to
`run_pipeline.py --update`, and *Start in* to `C:\Users\Gebruiker\ufc-predictor`.

## Linux / macOS cron

```cron
# m  h  dom mon dow  command
0    9  *   *   1    cd /home/me/ufc-predictor && .venv/bin/python run_pipeline.py --update --no-predict >> logs/cron.log 2>&1
0    19 *   *   3    cd /home/me/ufc-predictor && .venv/bin/python run_pipeline.py --update >> logs/cron.log 2>&1
```

cron mails or logs non-zero exits depending on your setup (`MAILTO=` at the top of the crontab).

## GitHub Actions (no machine needs to be on)

`data/` and `models/` are git-ignored, so a runner starts empty. That is fine: the whole
pipeline rebuilds from the Greco1899 source in about a minute (ingest downloads ~11 MB,
then clean → features → train). Two things must live in the repo instead of `data/raw/`:
- the upcoming card: commit it as `cards/upcoming_card.json` and pass `--card`;
- `ingest_overrides.csv` (already committed).

The workflow commits `outputs/` back, so prediction files and the results log persist
between runs, and tracking works across weeks.

`.github/workflows/weekly.yml`:

```yaml
name: weekly
on:
  schedule:
    - cron: "0 9 * * 1"    # Monday 09:00 UTC: results + tracking
    - cron: "0 19 * * 3"   # Wednesday 19:00 UTC: predictions
  workflow_dispatch: {}
permissions:
  contents: write
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.14"
          cache: pip
      - run: pip install -r requirements.txt
      - name: Run pipeline
        run: |
          if [ "${{ github.event.schedule }}" = "0 9 * * 1" ]; then
            python run_pipeline.py --update --no-predict
          else
            python run_pipeline.py --update --card cards/upcoming_card.json
          fi
      - name: Commit outputs
        if: always()
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add outputs/
          git diff --cached --quiet || git commit -m "chore: weekly pipeline outputs"
          git push
```

Notes:
- Cron times in Actions are UTC and may start several minutes late.
- GitHub disables scheduled workflows in a repository with no commits for 60 days; the
  weekly output commits keep it active.
- On a fresh runner there is no saved model, so the train stage always runs (≈10 s). Model
  files are not kept between runs; `model_version` in the outputs still records which
  model made each prediction.
- ESPN may or may not answer requests from GitHub's servers. The `--card` file makes that
  irrelevant; without `--card`, predict falls back to ESPN and fails clearly if it is blocked.
