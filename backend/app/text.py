import re
from collections import Counter

PAD = "<PAD>"
UNK = "<UNK>"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())

def build_vocab(texts, min_freq=1):
    counts = Counter(token for text in texts for token in tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for token, count in counts.items():
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def encode(text: str, vocab: dict, max_len=160):
    ids = [vocab.get(t, vocab[UNK]) for t in tokenize(text)[:max_len]]
    return ids or [vocab[UNK]]
