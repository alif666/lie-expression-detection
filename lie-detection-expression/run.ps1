param(
    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$Model = "models\baseline.joblib",
    [string]$OpenFaceBin = "OpenFace\FeatureExtraction.exe",
    [string]$OpenFaceCsv = "",
    [string]$OutJson = "outputs\alif_sentence_scores.json",
    [string]$OutCsv = "outputs\alif_sentence_scores.csv",
    [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Get-PythonExe {
    $venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { return $venvPython }
    return "python"
}

function Ensure-FfmpegOnPath {
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ff) { return }

    $candidate = "C:\Users\Asus\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
    if (Test-Path (Join-Path $candidate "ffmpeg.exe")) {
        $env:Path = "$candidate;$env:Path"
    }

    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $ff) {
        throw "ffmpeg not found. Install with: winget install Gyan.FFmpeg"
    }
}

$pythonExe = Get-PythonExe

if ($InstallDeps) {
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -e ".[dev]"
    & $pythonExe -m pip install openai-whisper
}

if (-not (Test-Path $Video)) {
    throw "Video not found: $Video"
}
if (-not (Test-Path $Model)) {
    throw "Model not found: $Model"
}

Ensure-FfmpegOnPath

if ([string]::IsNullOrWhiteSpace($OpenFaceCsv)) {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Video)
    $OpenFaceCsv = Join-Path "outputs\openface" "$stem.csv"
}

if (-not (Test-Path $OpenFaceCsv)) {
    if (-not (Test-Path $OpenFaceBin)) {
        throw "OpenFace binary not found: $OpenFaceBin"
    }
    $outDir = Split-Path -Path $OpenFaceCsv -Parent
    New-Item -ItemType Directory -Path $outDir -Force | Out-Null
    & $OpenFaceBin -f $Video -out_dir $outDir
    if (-not (Test-Path $OpenFaceCsv)) {
        throw "Expected OpenFace CSV not found after extraction: $OpenFaceCsv"
    }
}

New-Item -ItemType Directory -Path (Split-Path -Path $OutJson -Parent) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Path $OutCsv -Parent) -Force | Out-Null

& $pythonExe -m lie_detection_expression.score_cli `
    --video $Video `
    --openface-csv $OpenFaceCsv `
    --model $Model `
    --out-json $OutJson `
    --out-csv $OutCsv

Write-Host "Done."
Write-Host "JSON: $OutJson"
Write-Host "CSV:  $OutCsv"

