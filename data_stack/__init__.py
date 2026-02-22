"""The Stack dataset loader — multilingual code, large files, 1024-vocab BPE.

Streams bigcode/the-stack-dedup across multiple languages, keeps files >= MIN_FILE_CHARS,
trains a 1024-vocab BPE tokenizer, and caches token arrays as .npy files.
"""

import os
import sys

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
TRAIN_TOKENS_PATH = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_TOKENS_PATH = os.path.join(DATA_DIR, "val_tokens.npy")

MIN_FILE_CHARS = 32_000          # only keep large files — W needs long contexts
TRAIN_CHARS_TARGET = 32_000_000  # ~16M tokens at ~2x compression
VAL_CHARS_TARGET = 2_000_000
BPE_TRAIN_CHARS = 4_000_000
DEFAULT_VOCAB_SIZE = 1024

# Languages to sample from — diverse, high identifier-reuse
LANGUAGES = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]

sys.path.insert(0, os.path.dirname(DATA_DIR))
from data import BPETokenizer, DataLoader  # noqa: E402


def _stream_stack(char_target, seed=42):
    from datasets import load_dataset as hf_load, interleave_datasets

    print(f"Streaming The Stack ({', '.join(LANGUAGES)}, files >= {MIN_FILE_CHARS:,} chars)...")
    streams = []
    for lang in LANGUAGES:
        ds = hf_load(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{lang}",
            split="train",
            streaming=True,
        )
        streams.append(ds.select_columns(["content"]))

    combined = interleave_datasets(streams, seed=seed)

    chunks = []
    total = 0
    n_files = 0
    for row in combined:
        content = row["content"]
        if len(content) < MIN_FILE_CHARS:
            continue
        chunks.append(content)
        total += len(content)
        n_files += 1
        if n_files % 100 == 0:
            print(f"  {n_files} files, {total/1e6:.1f}M chars...", flush=True)
        if total >= char_target:
            break

    print(f"  {n_files} files, {total/1e6:.1f}M chars (done)")
    return "\n\n".join(chunks)


def load_dataset(vocab_size=DEFAULT_VOCAB_SIZE):
    os.makedirs(DATA_DIR, exist_ok=True)

    if (os.path.exists(TRAIN_TOKENS_PATH)
            and os.path.exists(VAL_TOKENS_PATH)
            and os.path.exists(TOKENIZER_PATH)):
        print("Loading cached stack tokenizer and token arrays...")
        tok = BPETokenizer.load(TOKENIZER_PATH)
        if tok.vocab_size == vocab_size:
            train_data = np.load(TRAIN_TOKENS_PATH)
            val_data = np.load(VAL_TOKENS_PATH)
            print(f"  Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
            return dict(train=train_data, val=val_data, vocab_size=vocab_size,
                        encode=tok.encode, decode=tok.decode)
        print(f"  vocab_size mismatch ({tok.vocab_size} vs {vocab_size}), retraining...")

    train_text = _stream_stack(TRAIN_CHARS_TARGET, seed=42)
    val_text = _stream_stack(VAL_CHARS_TARGET, seed=1337)

    bpe_sample = train_text[:BPE_TRAIN_CHARS]
    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(bpe_sample)/1e6:.1f}M chars...")
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(bpe_sample)
    tok.save(TOKENIZER_PATH)
    print(f"  Saved tokenizer to {TOKENIZER_PATH}")

    print("Tokenizing train set...")
    train_data = np.array(tok.encode(train_text), dtype=np.uint16)
    print(f"  {len(train_text)/1e6:.1f}M chars -> {len(train_data)/1e6:.1f}M tokens "
          f"({len(train_text)/len(train_data):.1f}x compression)")
    np.save(TRAIN_TOKENS_PATH, train_data)

    print("Tokenizing val set...")
    val_data = np.array(tok.encode(val_text), dtype=np.uint16)
    print(f"  {len(val_text)/1e6:.1f}M chars -> {len(val_data)/1e6:.1f}M tokens "
          f"({len(val_text)/len(val_data):.1f}x compression)")
    np.save(VAL_TOKENS_PATH, val_data)

    print(f"Cached token arrays to {DATA_DIR}")
    return dict(train=train_data, val=val_data, vocab_size=vocab_size,
                encode=tok.encode, decode=tok.decode)
