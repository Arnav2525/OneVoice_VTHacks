@echo off
setlocal
set "MODE=preview"
set "EXTRA="
for %%A in (%*) do (
  if /I "%%~A"=="live" set "MODE=live"
  if /I "%%~A"=="visible" set "EXTRA=-Visible"
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_onevoice.ps1" -Mode %MODE% %EXTRA%
