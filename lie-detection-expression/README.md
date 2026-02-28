# Lie Detection Expression

Beginner-friendly, PowerShell-first guide to run and understand this project end-to-end.

This project estimates **higher/lower deception probability** for each spoken sentence in a video by combining:
- OpenFace facial Action Unit (AU) features
- Whisper speech transcription with timestamps
- A baseline neural network classifier (scikit-learn MLP)

It does **not** prove whether a sentence is true/false. It outputs probabilistic risk signals.

## 1. Repository Layout After Clone
Your GitHub repository root contains:
- `docs/` (papers and analysis notes)
- `lie-detection-expression/` (actual runnable code)

So after cloning, always `cd` into the project folder:
```powershell
git clone https://github.com/alif666/lie-expression-detection.git
cd lie-expression-detection\lie-detection-expression
```

## 2. What This Project Produces
Final outputs:
- `outputs\alif_sentence_scores.csv`
- `outputs\alif_sentence_scores.json`

Each sentence row/object contains:
- `text`: transcribed sentence
- `start`: sentence start time in seconds
- `end`: sentence end time in seconds
- `deception_probability`: model score (0 to 1)
- `level`: `higher` if score >= 0.5, else `lower`

## 3. Full Setup On A New Windows PC (PowerShell)
Run all commands in PowerShell.

### Step 1: Open project folder
```powershell
cd C:\path\to\lie-expression-detection\lie-detection-expression
```

### Step 2: Create and activate virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If execution policy blocks scripts:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python -m pip install openai-whisper
```

### Step 4: Install FFmpeg (required by Whisper)
```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell, reactivate venv, then verify:
```powershell
cd C:\path\to\lie-expression-detection\lie-detection-expression
.\.venv\Scripts\Activate.ps1
ffmpeg -version
```

If still not found in current shell session, add temporary PATH:
```powershell
$env:Path = "C:\Users\Asus\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin;$env:Path"
ffmpeg -version
```

### Step 5: Install OpenFace
Place OpenFace folder in project root:
- `lie-detection-expression\openface\FeatureExtraction.exe`

Verify executable:
```powershell
cd .\openface
.\FeatureExtraction.exe -help
```

### Step 6: Download OpenFace model files (mandatory)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\download_models.ps1
```

Verify key model file:
```powershell
Test-Path .\model\patch_experts\cen_patches_0.25_of.dat
```
Expected: `True`

## 4. Run Pipeline (Manual, Explicit Steps)

### Step A: Generate OpenFace CSV from your video
```powershell
cd C:\path\to\lie-expression-detection\lie-detection-expression\openface
.\FeatureExtraction.exe -f "C:\path\to\lie-expression-detection\lie-detection-expression\sample_data\alif_sample_1.mp4" -out_dir "C:\path\to\lie-expression-detection\lie-detection-expression\outputs\openface"
```

Expected CSV:
- `outputs\openface\alif_sample_1.csv`

### Step B: Run sentence scoring
```powershell
cd C:\path\to\lie-expression-detection\lie-detection-expression
.\.venv\Scripts\Activate.ps1
python -m lie_detection_expression.score_cli --video sample_data\alif_sample_1.mp4 --openface-csv outputs\openface\alif_sample_1.csv --model models\baseline.joblib --out-json outputs\alif_sentence_scores.json --out-csv outputs\alif_sentence_scores.csv
```

### Step C: Inspect result
```powershell
Get-Content outputs\alif_sentence_scores.csv -TotalCount 20
```

## 5. Run Pipeline (One Command)
This uses the included automation script and auto-generates OpenFace CSV if missing.

```powershell
cd C:\path\to\lie-expression-detection\lie-detection-expression
.\run.ps1 -Video "sample_data\alif_sample_1.mp4"
```

Useful variants:
```powershell
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -OpenFaceBin "OpenFace\FeatureExtraction.exe"
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -OpenFaceCsv "outputs\openface\alif_sample_1.csv"
.\run.ps1 -Video "sample_data\alif_sample_1.mp4" -InstallDeps
```

## 6. How The Code Is Organized (Module By Module)

### `src\lie_detection_expression\config.py`
- `PipelineConfig`: central settings (confidence threshold, fps, window size, ANN hyperparams).

### `src\lie_detection_expression\data.py`
- `load_openface_csv(...)`:
  - reads OpenFace CSV
  - strips header whitespace (important for OpenFace exports)
  - checks required columns (`frame`, `timestamp`, `confidence`, `success`)
  - filters rows by confidence/success

### `src\lie_detection_expression\features.py`
- `build_frame_features(...)`:
  - discovers AU presence columns (`AUxx_c`)
  - discovers AU intensity columns (`AUxx_r`)
  - creates interaction features: `AUxx_c * AUxx_r`

### `src\lie_detection_expression\model.py`
- `train_model(...)`: builds scaler + MLP pipeline and trains it.
- `save_model(...)` / `load_model(...)`: model persistence via joblib.
- `TrainedModel`: wraps pipeline and feature-column metadata.

### `src\lie_detection_expression\train.py`
- `train_from_labeled_csv(...)`: end-to-end training from labeled OpenFace CSV.
- `train_and_save(...)`: train and write model artifact.

### `src\lie_detection_expression\inference.py`
- `aggregate_windows(...)`: clip-level score from fixed windows.
- `window_probabilities_with_times(...)`: window score + start/end time for alignment.

### `src\lie_detection_expression\service.py`
- `predict_clip_from_openface_csv(...)`: orchestrates load -> features -> align -> infer.

### `src\lie_detection_expression\openface.py`
- `resolve_openface_binary(...)`: finds OpenFace executable from args/env/PATH.
- `run_openface_extraction(...)`: executes FeatureExtraction and locates CSV output.

### `src\lie_detection_expression\sentence_scoring.py`
- `_transcribe_with_whisper(...)`: speech-to-text with timestamps.
- `_score_sentence(...)`: weighted overlap of sentence range vs AU windows.
- `score_video_sentences(...)`: complete sentence scoring pipeline.
- `write_sentence_scores(...)`: writes JSON + CSV outputs.

### `src\lie_detection_expression\cli.py` and `score_cli.py`
- command-line entrypoints for training and scoring.

### `src\lie_detection_expression\api\main.py`
- FastAPI endpoints:
  - `GET /health`
  - `POST /predict` (CSV to clip score)

### `tests\test_pipeline.py`
- tests for filtering, features, model roundtrip, window alignment, OpenFace resolution logic.

## 7. Dependencies Explained (What Each One Does)
From `pyproject.toml`:
- `numpy`: numeric arrays
- `pandas`: CSV/dataframe processing
- `scikit-learn`: MLP classifier and scaler
- `joblib`: model save/load
- `fastapi`: API framework
- `uvicorn`: API server
- `pytest` (dev): tests

External tools:
- `openai-whisper`: transcript + sentence timestamps
- `ffmpeg`: audio decoding for Whisper
- `OpenFace` (`FeatureExtraction.exe`): AU extraction from video frames

## 8. How This Implements The Research Pipeline
Mapping from analysis milestones/paper-style flow to code:

1. **Video -> OpenFace AUs**
   - done by OpenFace executable (`openface\FeatureExtraction.exe`)
   - output: frame-level CSV with `AU*_c`, `AU*_r`, `confidence`, `success`, `timestamp`

2. **Confidence gate / quality filter**
   - `load_openface_csv` in `data.py`
   - defaults: confidence >= 0.7 and success == 1

3. **Feature engineering**
   - `build_frame_features` in `features.py`
   - computes AU presence * intensity interactions

4. **ANN baseline**
   - `train_model` in `model.py`
   - MLP with ReLU + Adam + MinMax scaling

5. **Windowed inference**
   - `window_probabilities_with_times` and `aggregate_windows` in `inference.py`
   - converts frame features into window-level probabilities and clip average

6. **Sentence-level output**
   - `score_video_sentences` in `sentence_scoring.py`
   - aligns transcript timestamps (Whisper) with AU windows for per-sentence scores

## 9. Training Your Own Model
If you only have baseline model today, retrain with your own labeled data:

```powershell
python -m lie_detection_expression.cli --data data\sample_openface_labeled.csv --out models\baseline.joblib --max-epochs 60
```

Input training CSV must include:
- required OpenFace fields
- AU columns (`AU*_c`, `AU*_r`)
- label column (`label`, 0/1 by default)

## 10. Common Errors And Fixes

### `'source' is not recognized`
You are in Windows shell. Use:
```powershell
.\.venv\Scripts\Activate.ps1
```

### `download_models.ps1 ... exists in current location`
Use:
```powershell
.\download_models.ps1
```

### `Could not find CEN patch experts`
OpenFace models missing. Run `.\download_models.ps1`.

### `FileNotFoundError [WinError 2]` during Whisper
`ffmpeg` missing from current PATH. Reopen shell and verify `ffmpeg -version`.

### `Missing required columns: confidence, success, timestamp`
Use latest code and reinstall:
```powershell
python -m pip install -e .[dev]
```

### `FP16 is not supported on CPU; using FP32 instead`
Normal warning on CPU-only machines.

## 11. Basic Verification Checklist
Before running full scoring, ensure all are true:
1. `.\.venv\Scripts\Activate.ps1` works
2. `python -m pip install -e .[dev]` completed
3. `ffmpeg -version` works
4. `.\openface\FeatureExtraction.exe -help` works
5. `Test-Path .\openface\model\patch_experts\cen_patches_0.25_of.dat` is `True`
6. `outputs\openface\alif_sample_1.csv` exists after extraction

## 12. Ethical Note
This system provides probabilistic behavior cues, not factual lie detection. Do not use it as sole evidence for high-stakes decisions.
