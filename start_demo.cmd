@echo off
setlocal
cd /d "%~dp0"

set MODE=--live
if /i "%~1"=="preview" set MODE=--preview

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js is required for the story page. Install it from https://nodejs.org and run this again.
  pause
  exit /b 1
)
where python >nul 2>&1
if errorlevel 1 (
  echo Python is required for the live app. Activate the project environment and run this again.
  pause
  exit /b 1
)

echo Starting the story page on http://127.0.0.1:4319/ and the live app on http://127.0.0.1:8771/
if /i "%MODE%"=="--preview" (
  echo Preview mode: no camera or microphone will be opened.
) else (
  echo The camera and microphone open only after you press Start listening in the app.
)

start "One Voice story" /min cmd /c "cd /d "%~dp0one-voice-working-copy" && node server.cjs"
start "One Voice app" cmd /k "cd /d "%~dp0" && python -m demo.tap_to_select %MODE% --port 8771 --no-browser"

timeout /t 4 /nobreak >nul
start "" http://127.0.0.1:4319/
endlocal
