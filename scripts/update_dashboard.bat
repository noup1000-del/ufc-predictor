@echo off
rem Refresh the online dashboard after a card change (bouts added, cancelled or replaced).
rem   1. run_pipeline.py --predict-all : re-fetch every card from ufc.com, re-predict, rebuild index.html
rem   2. commit outputs/ (predictions, results reviews, tracking log; only if something changed)
rem   3. push to main, which triggers .github/workflows/pages.yml -> GitHub Pages
rem Uses the local data and models/latest.pkl (no ingest or training). Exit code != 0 on failure.
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" run_pipeline.py --predict-all
if errorlevel 1 (
    echo Prediction run failed; nothing committed. See logs\pipeline.log
    exit /b 1
)

git add -- outputs
git diff --cached --quiet -- outputs
if not errorlevel 1 (
    echo No card changes since the last publish; the dashboard is already current.
    exit /b 0
)

set STAMP=
for /f "delims=" %%d in ('.venv\Scripts\python.exe -c "import datetime as d;print(d.datetime.now(d.timezone.utc).strftime('%%Y-%%m-%%d %%H:%%M UTC'))"') do set "STAMP=%%d"
git commit -q -m "predictions: refresh scheduled cards (%STAMP%)" -- outputs
if errorlevel 1 (
    echo git commit failed.
    exit /b 1
)

git push origin main
if errorlevel 1 (
    echo git push failed ^(is main behind origin? run "git pull --rebase origin main" and push again^).
    exit /b 1
)
echo Pushed. GitHub Pages redeploys in about a minute: https://noup1000-del.github.io/ufc-predictor/
exit /b 0
