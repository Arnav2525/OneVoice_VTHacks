
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$CodeStaging = Join-Path $env:TEMP "onevoice-kaggle-code"
if (Test-Path $CodeStaging) { Remove-Item $CodeStaging -Recurse -Force }
New-Item -ItemType Directory -Path $CodeStaging | Out-Null

$RootItems = @(
    "pyproject.toml",
    "src",
    "bench",
    "configs"
)
foreach ($item in $RootItems) {
    $src = Join-Path $Root $item
    if (-not (Test-Path $src)) { throw "missing $item" }
    Copy-Item -Path $src -Destination $CodeStaging -Recurse -Force
}

$ScriptDir = Join-Path $CodeStaging "scripts"
New-Item -ItemType Directory -Path $ScriptDir | Out-Null
$ScriptFiles = @(
    "vendor_dolphin.py",
    "export_dolphin_separation.py",
    "vendor_realtse.py",
    "fetch_realtse_checkpoints.py",
    "verify_retinaface_alignment.py",
    "verify_lip_roi_real_crop.py",
    "replay_harness.py",
    "kaggle_replay_harness_cell.py",
    "kaggle_retinaface_quality_cell.py",
    "kaggle_retinaface_timing_cell.py",
    "kaggle_gpu_contention_cell.py",
    "kaggle_liproi_window2s_cell.py",
    "measure_gpu_contention.py",
    "time_retinaface_gpu.py",
    "make_ood_mixture.py",
    "ood_replay_check.py"
)
foreach ($name in $ScriptFiles) {
    $src = Join-Path $Root "scripts\$name"
    if (-not (Test-Path $src)) { throw "missing scripts/$name" }
    Copy-Item -Path $src -Destination (Join-Path $ScriptDir $name) -Force
    if ($name -in @("vendor_dolphin.py", "export_dolphin_separation.py", "vendor_realtse.py", "fetch_realtse_checkpoints.py")) {
        Copy-Item -Path $src -Destination (Join-Path $CodeStaging $name) -Force
    }
}

$CodeZip = Join-Path $Root "onevoice-kaggle-code.zip"
if (Test-Path $CodeZip) { Remove-Item $CodeZip -Force }
python (Join-Path $Root "scripts/_zip_for_kaggle.py") $CodeStaging $CodeZip
if ($LASTEXITCODE -ne 0) { throw "code zip failed" }

$DataStaging = Join-Path $env:TEMP "onevoice-kaggle-tier-a"
if (Test-Path $DataStaging) { Remove-Item $DataStaging -Recurse -Force }
New-Item -ItemType Directory -Path $DataStaging | Out-Null

$tt = Join-Path $Root "data\dolphin_tier_a\tt"
$mouth = Join-Path $Root "data\dolphin_tier_a\mouth_cache"
if (-not (Test-Path (Join-Path $tt "mix.json"))) {
    throw "missing data/dolphin_tier_a/tt; run prepare_dolphin_tier_a_data.py first"
}
if (-not (Test-Path $mouth)) {
    throw "missing data/dolphin_tier_a/mouth_cache (needed by RetinaFace quality cell)"
}
Copy-Item -Path $tt -Destination (Join-Path $DataStaging "tt") -Recurse -Force
Copy-Item -Path $mouth -Destination (Join-Path $DataStaging "mouth_cache") -Recurse -Force

$DataZip = Join-Path $Root "onevoice-dolphin-tier-a.zip"
if (Test-Path $DataZip) { Remove-Item $DataZip -Force }
python (Join-Path $Root "scripts/_zip_for_kaggle.py") $DataStaging $DataZip
if ($LASTEXITCODE -ne 0) { throw "data zip failed" }

$codeMb = [math]::Round((Get-Item $CodeZip).Length / 1MB, 2)
$dataMb = [math]::Round((Get-Item $DataZip).Length / 1MB, 2)
Write-Host "Created (Kaggle-safe forward-slash paths):"
Write-Host "  $CodeZip  (${codeMb} MB) -> Kaggle dataset onevoice-kaggle-code / onevoice-code"
Write-Host "  $DataZip  (${dataMb} MB) -> Kaggle dataset onevoice-dolphin-tier-a / dolphin-tiera"
Write-Host "    includes tt/ + mouth_cache/"
Write-Host ""
Write-Host "No GitHub required."
