"""Unified dataset loader for hebbian-language.

Supports three datasets:
    - "pg19"  : Project Gutenberg long-form prose (emozilla/pg19)
    - "code_parrot" : Python code from codeparrot-clean
    - "the_stack"   : Multilingual code from The Stack (bigcode/the-stack-dedup)

Usage:
    from data import load_dataset, DataLoader

    ds = load_dataset("pg19")          # or "code_parrot" or "the_stack"
    loader = DataLoader(ds["train"], batch_size=32, seq_len=256)
    x, y = loader.batch()
"""

import json
import os

from datasets import load_dataset as hf_load, interleave_datasets
import numpy as np
import torch

class BPETokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}

    def train(self, text: str):
        data = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = self.vocab_size - 256
        ids = list(data)
        for i in range(num_merges):
            counts = {}
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break
            pair = max(counts, key=lambda p: counts[p])
            new_id = 256 + i
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            ids = self._merge(ids, pair, new_id)
            if (i + 1) % 32 == 0:
                print(f"  BPE merge {i + 1}/{num_merges}: {self.vocab[new_id]!r} (freq={counts[pair]})")

    @staticmethod
    def _merge(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for pair_idx, pair in enumerate(self.merges):
            ids = self._merge(ids, pair, 256 + pair_idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            obj = json.load(f)
        tok = cls(vocab_size=obj["vocab_size"])
        tok.merges = [tuple(p) for p in obj["merges"]]
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(tok.merges):
            tok.vocab[256 + i] = tok.vocab[a] + tok.vocab[b]
        return tok


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Simple random-batch data loader from a flat token array."""

    def __init__(self, data: np.ndarray, batch_size: int, seq_len: int):
        self.data = data
        self.B = batch_size
        self.T = seq_len

    def batch(self):
        ix = torch.randint(len(self.data) - self.T, (self.B,))
        x = torch.stack([torch.from_numpy(self.data[i : i + self.T].copy()) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.T].copy()) for i in ix])
        return x, y


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "pg19": dict(
        cache_dir=os.path.join(DATA_DIR, "pg19"),
        default_vocab_size=1024,
        train_chars=80_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        min_file_chars=0,
    ),
    "code_parrot": dict(
        cache_dir=os.path.join(DATA_DIR, "codeparrot"),
        default_vocab_size=1024,
        train_chars=64_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        min_file_chars=4096,
    ),
    "the_stack": dict(
        cache_dir=os.path.join(DATA_DIR, "the_stack"),
        default_vocab_size=1024,
        train_chars=64_000_000,
        val_chars=4_000_000,
        bpe_train_chars=5_000_000,
        min_file_chars=32_000,
    ),
}

STACK_LANGUAGES = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_dataset(name="pg19", vocab_size=None):
    """Load (or download + tokenize + cache) a dataset.

    Args:
        name: One of "pg19", "code_parrot", "the_stack".
        vocab_size: BPE vocab size. Defaults to the dataset's default.

    Returns:
        dict with keys: train, val, vocab_size, encode, decode.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {list(DATASETS)}")

    cfg = DATASETS[name]
    cache_dir = str(cfg["cache_dir"])
    if vocab_size is None:
        vocab_size = cfg["default_vocab_size"]

    tok_path = os.path.join(cache_dir, "tokenizer.json")
    train_path = os.path.join(cache_dir, "train_tokens.npy")
    val_path = os.path.join(cache_dir, "val_tokens.npy")

    os.makedirs(cache_dir, exist_ok=True)

    # Try loading from cache
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(tok_path):
        print(f"Loading cached {name} tokenizer and token arrays...")
        tok = BPETokenizer.load(tok_path)
        if tok.vocab_size == vocab_size:
            train_data = np.load(train_path)
            val_data = np.load(val_path)
            print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
            return dict(train=train_data, val=val_data, vocab_size=vocab_size,
                        encode=tok.encode, decode=tok.decode)
        print(f"  vocab_size mismatch ({tok.vocab_size} vs {vocab_size}), retraining...")

    # Stream text
    train_text, val_text = _get_texts(name, cfg)

    # Train tokenizer
    bpe_sample = train_text[:cfg["bpe_train_chars"]]
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(bpe_sample):,} chars...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(bpe_sample)
    tok.save(tok_path)
    print(f"  Saved tokenizer to {tok_path}")

    # Tokenize and cache
    print("Tokenizing train set...")
    train_data = np.array(tok.encode(train_text), dtype=np.uint16)
    print(f"  {len(train_text):,} chars -> {len(train_data):,} tokens ({len(train_text)/len(train_data):.1f}x compression)")
    np.save(train_path, train_data)

    print("Tokenizing val set...")
    val_data = np.array(tok.encode(val_text), dtype=np.uint16)
    print(f"  {len(val_text):,} chars -> {len(val_data):,} tokens ({len(val_text)/len(val_data):.1f}x compression)")
    np.save(val_path, val_data)

    print(f"Cached token arrays to {cache_dir}")
    return dict(train=train_data, val=val_data, vocab_size=vocab_size,
                encode=tok.encode, decode=tok.decode)

def _get_texts(name, cfg):
    """Download/stream train and val text for a dataset."""
    if name == "pg19":
        print("Streaming PG-19 train books...")
        train_text = _stream_pg19("train", cfg["train_chars"])
        print("Streaming PG-19 validation books...")
        val_text = _stream_pg19("validation", cfg["val_chars"])
    elif name == "code_parrot":
        print("Streaming codeparrot-clean Python (train)...")
        train_text = _stream_code(cfg["train_chars"], seed=42)
        print("Streaming codeparrot-clean Python (val)...")
        val_text = _stream_code(cfg["val_chars"], seed=1337)
    elif name == "the_stack":
        train_text = _stream_stack(cfg["train_chars"], seed=42)
        val_text = _stream_stack(cfg["val_chars"], seed=1337)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {list(DATASETS)}")
    return train_text, val_text


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _stream_pg19(split, char_target):
    ds = hf_load("emozilla/pg19", split=split, streaming=True)
    texts = (row["text"] for row in ds)
    return _collect_chunks(texts, char_target, label=f"{split}", unit="books", log_every=50)


def _stream_code(char_target, seed=None):
    ds = hf_load("codeparrot/codeparrot-clean", split="train", streaming=True)
    if seed is not None:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
    min_chars = int(DATASETS["code_parrot"]["min_file_chars"])
    texts = _filter_by_length(ds, "content", min_chars)
    return _collect_chunks(texts, char_target, label="", unit="files", log_every=200)


def _stream_stack(char_target, seed=42):
    print(f"Streaming The Stack ({', '.join(STACK_LANGUAGES)}, files >= {int(DATASETS['the_stack']['min_file_chars']):,} chars)...")
    streams = []
    for lang in STACK_LANGUAGES:
        ds = hf_load("bigcode/the-stack-dedup", data_dir=f"data/{lang}", split="train", streaming=True)
        streams.append(ds.select_columns(["content"]))
    combined = interleave_datasets(streams, seed=seed)
    min_chars = int(DATASETS["the_stack"]["min_file_chars"])
    texts = _filter_by_length(combined, "content", min_chars)
    return _collect_chunks(texts, char_target, label="", unit="files", log_every=100, fmt_total=lambda t: f"{t/1e6:.1f}M")


def _collect_chunks(texts, char_target, *, label="", unit="items", log_every=100, fmt_total=None):
    """Accumulate text chunks from an iterator until char_target is reached."""
    if fmt_total is None:
        fmt_total = lambda t: f"{t:,}"
    prefix = f"  {label}: " if label else "  "
    chunks, total, n = [], 0, 0
    for text in texts:
        chunks.append(text)
        total += len(text)
        n += 1
        if n % log_every == 0:
            print(f"{prefix}{n} {unit}, {fmt_total(total)} chars...", flush=True)
        if total >= char_target:
            break
    print(f"{prefix}{n} {unit}, {fmt_total(total)} chars (done)")
    return "\n\n".join(chunks)


def _filter_by_length(dataset, field, min_chars):
    """Yield text values from dataset rows that meet the minimum length."""
    for row in dataset:
        if len(row[field]) >= min_chars:
            yield row[field]


