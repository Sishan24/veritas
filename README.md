# VERITAS — The Claim & Context Intelligence Engine

A runnable student research-project starter for evidence-aware NLP:
- React + Vite frontend with a persistent dark/light theme toggle
- FastAPI backend
- PyTorch BiLSTM + attention claim/evidence classifier
- CSV-based training pipeline
- Transparent demo mode when no trained checkpoint exists

## Important research note
The included `data/sample_claims.csv` is a tiny **illustrative smoke-test dataset**, not a research-quality dataset. Predictions before training are explicitly marked `demo_mode`. Do not report demo predictions as model results. For meaningful experiments, replace/expand the CSV using properly licensed, documented datasets such as FEVER, LIAR, or AVeriTeC, and preserve their train/validation/test splits.

## Requirements
- Python 3.12
- Node.js 20+ (or a recent LTS release)
- npm

## 1. Start the backend (Windows PowerShell)
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Backend: http://127.0.0.1:8000  
API docs: http://127.0.0.1:8000/docs

If `py -3.12` is not recognized but `python --version` reports Python 3.12, use:
```powershell
python -m venv .venv
```

## 2. Start the frontend (second terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open the local URL printed by Vite (normally http://localhost:5173).

## 3. Train the PyTorch model
Stop neither server; open a third terminal:
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m app.train --data data/sample_claims.csv --epochs 25
```
This writes `backend/artifacts/veritas_bilstm.pt`. Restart the backend after training. The tiny sample is only for checking that the training pipeline works; it is not enough for credible evaluation.

CSV columns:
`claim,evidence,label`
Labels: `SUPPORTED`, `REFUTED`, `NOT_ENOUGH_INFO`.

## Project structure
```text
VERITAS/
  backend/
    app/
      main.py          FastAPI endpoints
      model.py         PyTorch BiLSTM + attention
      text.py          tokenizer and vocabulary
      train.py         training loop
    data/sample_claims.csv
    requirements.txt
  frontend/
    src/
      App.jsx
      main.jsx
      styles.css
    package.json
    index.html
    vite.config.js
  .gitignore
```

## Research extensions to implement
1. Add a proper evidence retriever (BM25 or dense retrieval) over a document collection.
2. Evaluate retrieval separately using Recall@K / Precision@K.
3. Compare MLP, CNN, LSTM, BiLSTM and BiLSTM+attention under identical splits.
4. Add optimizer, activation, dropout and embedding ablations.
5. Report Macro-F1, per-class precision/recall, confusion matrix, and latency.
6. Add evidence highlighting only when spans are actually aligned to source text.

## Current scope
The app accepts one claim and one evidence passage. It classifies the *relationship between those supplied texts*. It does not independently verify the real-world truth of a claim or browse the internet for sources.
