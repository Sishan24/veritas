import torch
from torch import nn

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]

class ClaimEvidenceBiLSTMAttention(nn.Module):
    \"\"\"BiLSTM encoder with learned attention over the concatenated pair tokens.\"\"\"
    def __init__(self, vocab_size: int, embedding_dim=96, hidden_dim=96, num_classes=3, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True,
            bidirectional=True, dropout=0.0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, token_ids, lengths=None):
        embedded = self.embedding(token_ids)
        encoded, _ = self.encoder(embedded)
        scores = self.attention(encoded).squeeze(-1)
        if lengths is not None:
            positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
            mask = positions >= lengths.unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.dropout(pooled)), weights
