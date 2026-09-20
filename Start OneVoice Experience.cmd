@echo off
setlocal
cd /d "%~dp0"
set "ONEVOICE_MODE=--preview"
if /I "%~1"=="live" set "ONEVOICE_MODE=--live"
echo Starting the local listening app in %ONEVOICE_MODE% mode.
echo Camera and microphone remain off until Start listening is pressed.
start "OneVoice app" cmd /k python -m demo.tap_to_select %ONEVOICE_MODE% --port 8771 --no-browser
cd /d "%~dp0one-voice-working-copy"
start "OneVoice story" cmd /k node server.cjs
timeout /t 2 /nobreak >nul
echo Opening the 3D story at http://127.0.0.1:4319/
start "" "http://127.0.0.1:4319/"
