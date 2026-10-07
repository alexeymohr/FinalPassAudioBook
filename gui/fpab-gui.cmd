@echo off
rem Launch the FinalPass AudioBook GUI on Windows.
rem Uses the `py` launcher if present, else `python`. For no console window, run:
rem   pythonw -m fpab_gui
setlocal
set "HERE=%~dp0"
if defined PYTHONPATH (set "PYTHONPATH=%HERE%;%PYTHONPATH%") else (set "PYTHONPATH=%HERE%")
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -m fpab_gui %*
) else (
    python -m fpab_gui %*
)
