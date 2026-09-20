param(
    [ValidateSet('preview', 'live')][string]$Mode = 'preview',
    [switch]$Visible,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null

$appUrl = 'http://127.0.0.1:8771/api/state'
$storyUrl = 'http://127.0.0.1:4319/'
$style = if ($Visible) { 'Normal' } else { 'Hidden' }

function Test-Up([string]$url) {
    try { Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

function Show-Problem([string]$text) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show($text, 'OneVoice') | Out-Null
}

foreach ($tool in 'python', 'node') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Show-Problem "$tool was not found. Install it or activate the project environment, then start OneVoice again."
        exit 1
    }
}

if (-not (Test-Up $appUrl)) {
    Start-Process -FilePath 'python' `
        -ArgumentList '-m', 'demo.tap_to_select', "--$Mode", '--port', '8771', '--no-browser' `
        -WorkingDirectory $root -WindowStyle $style `
        -RedirectStandardOutput (Join-Path $root 'logs\app.log') `
        -RedirectStandardError (Join-Path $root 'logs\app.err.log')
}
if (-not (Test-Up $storyUrl)) {
    Start-Process -FilePath 'node' -ArgumentList 'server.cjs' `
        -WorkingDirectory (Join-Path $root 'one-voice-working-copy') -WindowStyle $style `
        -RedirectStandardOutput (Join-Path $root 'logs\story.log') `
        -RedirectStandardError (Join-Path $root 'logs\story.err.log')
}

$deadline = (Get-Date).AddSeconds(150)
while ((Get-Date) -lt $deadline) {
    if ((Test-Up $appUrl) -and (Test-Up $storyUrl)) { break }
    Start-Sleep -Milliseconds 500
}

if (-not ((Test-Up $appUrl) -and (Test-Up $storyUrl))) {
    Show-Problem "OneVoice did not start in time. Details are in the logs folder: $root\logs"
    exit 1
}

if (-not $NoBrowser) { Start-Process $storyUrl }
