@echo off
REM Simple wrapper to run the PowerShell label sync script.
REM Usage: sync_labels.bat [labels-file]

set "LABEL_FILE=%~1"
if "%LABEL_FILE%"=="" (
    set "LABEL_FILE=labels.txt"
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_labels.ps1" -LabelFile "%LABEL_FILE%"
