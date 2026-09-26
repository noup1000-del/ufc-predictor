@echo off
rem After an event (results usually reach the data source within ~2 days):
rem   1. run_pipeline.py --update --no-predict : import results, track picks, write the review
rem      (outputs/reviews/), retrain if there is new data (promotion gate applies)
rem   2. update_dashboard.bat : re-predict upcoming cards and publish, incl. the Results tab
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" run_pipeline.py --update --no-predict
if errorlevel 1 (
    echo Results/tracking run failed; nothing published. See logs\pipeline.log
    exit /b 1
)
call "%~dp0update_dashboard.bat"
exit /b %ERRORLEVEL%
