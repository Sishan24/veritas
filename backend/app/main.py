from pathlib import Path
import json
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import ClaimEvidenceBiLSTMAttention, LABELS
from .text import encode

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CHECKPOINT = ARTIFACTS / "veritas_bilstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="VERITAS API", version="1.0.0",
              description="Evidence-aware claim/evidence relationship classifier.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class AnalyzeRequest(BaseModel):
    claim: str = Field(min_length=8, max_length=2000)
    evidence: str = Field(min_length=8, max_length=5000)

model = None
vocab = None

def load_checkpoint():
    global model, vocab
    if not CHECKPOINT.exists():
        model, vocab = None, None
        return
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    vocab = checkpoint["vocab"]
    model = ClaimEvidenceBiLSTMAttention(len(vocab)).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

@app.on_event("startup")
def startup():
    load_checkpoint()

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "mode": "trained_model" if model is not None else "demo_mode"}

@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest):
    if model is None:
        # Deliberately non-assertive fallback: no untrained neural prediction is presented as fact.
        return {
            "label": "NOT_ENOUGH_INFO",
            "confidence": None,
            "mode": "demo_mode",
            "summary": "The project is running, but no trained checkpoint is loaded yet.",
            "explanation": "Train the BiLSTM-attention model with the supplied CSV pipeline, then restart the API. This demo response is not a factual assessment.",
            "claim": payload.claim,
            "evidence": payload.evidence,
            "attention": []
        }

    joined = payload.claim + " [SEP] " + payload.evidence
    ids = encode(joined, vocab)
    tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits, attention = model(tensor, lengths)
        probabilities = torch.softmax(logits, dim=-1)[0]
        index = int(torch.argmax(probabilities).item())
    return {
        "label": LABELS[index],
        "confidence": round(float(probabilities[index].item()), 4),
        "mode": "trained_model",
        "summary": "Predicted relationship between the supplied claim and evidence.",
        "explanation": "This is a learned model prediction, not an independent determination of real-world truth.",
        "claim": payload.claim,
        "evidence": payload.evidence,
        "attention": [round(float(x), 4) for x in attention[0].cpu().tolist()[:40]]
    }
