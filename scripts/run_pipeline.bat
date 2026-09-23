@echo off
rem Run the pipeline with the project's .venv, from any working directory.
rem Arguments are passed through, e.g.:  run_pipeline.bat --update --no-predict
rem The exit code of run_pipeline.py is returned, so Task Scheduler shows failures.
cd /d "%~dp0.."
".venv\Scripts\python.exe" run_pipeline.py %*
exit /b %ERRORLEVEL%
