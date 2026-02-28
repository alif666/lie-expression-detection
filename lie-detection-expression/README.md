# Lie Detection Expression

PowerShell-first guide to run sentence-level deception scoring on a video using:
- OpenFace (`FeatureExtraction.exe`)
- Whisper transcription
- This project’s scoring pipeline

This guide is based on real setup issues encountered during execution.

## What You Get
Given a video, output files include one row per sentence:
- `text`
- `start`
- `end`
- `deception_probability`
- `level` (`higher` if `>= 0.5`, else `lower`)

Output paths:
- `outputs\alif_sentence_scores.csv`
- `outputs\alif_sentence_scores.json`

## 1. Open PowerShell In Project Root
```powershell
cd C:\projects\lie-detection-expression
```

## 2. Create And Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install Python Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install openai-whisper
```

## 4. Install FFmpeg (Required By Whisper)
```powershell
winget install Gyan.FFmpeg
```

Important: restart PowerShell after install, then reactivate venv.
```powershell
cd C:\projects\lie-detection-expression
.\.venv\Scripts\Activate.ps1
ffmpeg -version
```

If `ffmpeg` is still not found in current session:
```powershell
$env:Path = "C:\Users\Asus\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin;$env:Path"
ffmpeg -version
```

## 5. Install OpenFace In Project Folder
Expected folder:
- `C:\projects\lie-detection-expression\openface`

Verify binary:
```powershell
cd C:\projects\lie-detection-expression\openface
.\FeatureExtraction.exe -help
```

## 6. Download OpenFace Models (Mandatory)
Without this, OpenFace fails with missing `cen_patches_0.25_of.dat`.

From PowerShell in `openface` folder:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\download_models.ps1
```

Verify model file:
```powershell
Test-Path .\model\patch_experts\cen_patches_0.25_of.dat
```
Must return `True`.

## 7. Generate OpenFace CSV From Video
```powershell
cd C:\projects\lie-detection-expression\openface
.\FeatureExtraction.exe -f "C:\projects\lie-detection-expression\sample_data\alif_sample_1.mp4" -out_dir "C:\projects\lie-detection-expression\outputs\openface"
```

Expected CSV:
- `C:\projects\lie-detection-expression\outputs\openface\alif_sample_1.csv`

## 8. Run Sentence Scoring
Back in project root:
```powershell
cd C:\projects\lie-detection-expression
.\.venv\Scripts\Activate.ps1
python -m lie_detection_expression.score_cli --video sample_data\alif_sample_1.mp4 --openface-csv outputs\openface\alif_sample_1.csv --model models\baseline.joblib --out-json outputs\alif_sentence_scores.json --out-csv outputs\alif_sentence_scores.csv
```

## 8B. Automation Script (`run.ps1`)
Runs extraction (if CSV missing) + sentence scoring in one command.

From project root:
```powershell
.\run.ps1 -Video "sample_data\alif_sample_1.mp4"
```

With explicit OpenFace path:
```powershell
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -OpenFaceBin "OpenFace\FeatureExtraction.exe"
```

With explicit CSV (skip extraction):
```powershell
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -OpenFaceCsv "outputs\openface\alif_sample_1.csv"
```

Install/update dependencies via script:
```powershell
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -InstallDeps
```

## 9. Inspect Results
```powershell
Get-Content outputs\alif_sentence_scores.csv -TotalCount 20
```

## One-Command Variant (Auto OpenFace Extraction)
You can skip `--openface-csv` and let CLI run OpenFace automatically:
```powershell
python -m lie_detection_expression.score_cli --video sample_data\alif_sample_1.mp4 --openface-bin "openface\FeatureExtraction.exe" --model models\baseline.joblib --out-json outputs\alif_sentence_scores.json --out-csv outputs\alif_sentence_scores.csv
```

## Troubleshooting

### `'source' is not recognized`
You are in Windows shell. Use:
```powershell
.\.venv\Scripts\Activate.ps1
```

### `download_models.ps1 was not found ... exists in current location`
Run with `.\` prefix:
```powershell
.\download_models.ps1
```

### `A positional parameter cannot be found ... Set-ExecutionPolicy`
You merged two commands. Run separately:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\download_models.ps1
```

### `Could not find CEN patch experts`
Model files are missing. Run `.\download_models.ps1` and verify:
```powershell
Test-Path .\model\patch_experts\cen_patches_0.25_of.dat
```

### `FileNotFoundError [WinError 2]` during Whisper
`ffmpeg` is not available to Python in current shell.
1. Reopen PowerShell.
2. Reactivate venv.
3. Run `ffmpeg -version`.

### `Missing required columns: confidence, success, timestamp`
Use latest code from this repo. Loader now strips OpenFace header spaces automatically.
Reinstall editable package in venv:
```powershell
python -m pip install -e .[dev]
```

### Command broke across lines
Use one line, or PowerShell backticks for line continuation.

## Notes
- `FP16 is not supported on CPU; using FP32 instead` is normal on CPU.
- Scores are probabilistic signals, not proof of truth/lying.
