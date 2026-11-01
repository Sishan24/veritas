import argparse
import csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .model import ClaimEvidenceBiLSTMAttention, LABELS
from .text import build_vocab, encode

class PairDataset(Dataset):
    def __init__(self, rows, vocab, max_len=160):
        self.items = []
        for row in rows:
            text = row["claim"].strip() + " [SEP] " + row["evidence"].strip()
            ids = encode(text, vocab, max_len)
            self.items.append((ids, LABELS.index(row["label"].strip().upper())))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max())
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        ids[i, :len(seq)] = torch.tensor(seq)
    return ids, lengths, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_claims.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = Path(__file__).resolve().parents[1] / data_path
    with data_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"claim", "evidence", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("CSV must contain claim,evidence,label columns and at least one row.")
    for row in rows:
        if row["label"].strip().upper() not in LABELS:
            raise ValueError(f"Unknown label: {row['label']}. Expected {LABELS}")

    texts = [r["claim"] + " [SEP] " + r["evidence"] for r in rows]
    vocab = build_vocab(texts)
    dataset = PairDataset(rows, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClaimEvidenceBiLSTMAttention(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(dataset)} rows | vocab={len(vocab)} | device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        print(f"Epoch {epoch:02d}/{args.epochs} | loss={total_loss/seen:.4f} | train_acc={correct/seen:.3f}")

    out = Path(__file__).resolve().parents[1] / "artifacts"
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "vocab": vocab}, out / "veritas_bilstm.pt")
    print(f"Saved checkpoint: {out / 'veritas_bilstm.pt'}")
    print("Reminder: training accuracy on a tiny sample is not research evaluation.")

if __name__ == "__main__":
    main()
