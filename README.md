# Grammar Scoring Engine — Ready-to-Run Demo

This is a self-contained VS Code project with **100 training + 100 test WAV samples**.

## Important dataset note

The bundled audio is **synthetic demo speech generated locally with eSpeak**, and the training scores are **proxy/demo grammar levels** (20 samples for each score 1–5). They are included so the project is genuinely runnable without downloading a 769-sample competition dataset.

These scores are **not the official competition labels** and should not be submitted to the competition as if they were. Replace `train.csv`, `test.csv`, `sample_submission.csv`, and `audio/` with the real competition data for a real leaderboard submission.

## Structure

```text
grammar-scoring-full/
├── grammar_scoring_engine.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── requirements.txt
├── train_model.py
├── app.py
├── run_demo.py
├── README.md
└── audio/
    ├── train_001.wav ... train_100.wav
    └── test_001.wav  ... test_100.wav
```

## Run in VS Code

### 1. Create environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the complete demo

```powershell
python run_demo.py
```

This trains the model, prints RMSE/Pearson, and creates `submission.csv`.

### 4. Run the notebook

Open `grammar_scoring_engine.ipynb`, select `.venv` as the kernel, and run all cells.

### 5. Run the API

First train the model:

```powershell
python train_model.py
```

Then:

```powershell
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The API accepts an audio file and returns a grammar score from 0–5. It uses Faster-Whisper when an arbitrary audio file is uploaded.

## Competition mode

For the real competition dataset, replace the demo files with the real data. The notebook can use a transcript column if present; otherwise it can transcribe WAV files with Faster-Whisper.
