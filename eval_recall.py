"""Synthetic recall: can Hebbian memory carry associations across a Mamba reset?

Sequence: [k0 v0 k1 v1 ... | kπ(0) vπ(0) kπ(1) vπ(1) ...]
           store phase        query phase (Mamba reset, only W persists)

No filler. Loss only on recall positions.
Without memory: chance = 1/16. With memory: should reach ~100%.
"""

import torch
import torch.nn.functional as F
from model import Config, HebbianMamba

NUM_KEYS = 16
NUM_VALS = 16
VOCAB = NUM_KEYS + NUM_VALS
PAIRS = 8
SEQ_LEN = 4 * PAIRS  # exactly pairs + queries, no waste
MID = SEQ_LEN // 2


def make_batch(B, device):
    x = torch.zeros(B, SEQ_LEN, dtype=torch.long, device=device)

    for b in range(B):
        keys = torch.randperm(NUM_KEYS, device=device)[:PAIRS]
        vals = torch.randint(NUM_KEYS, VOCAB, (PAIRS,), device=device)

        for i in range(PAIRS):
            x[b, 2 * i] = keys[i]
            x[b, 2 * i + 1] = vals[i]

        perm = torch.randperm(PAIRS, device=device)
        for i in range(PAIRS):
            x[b, MID + 2 * i] = keys[perm[i]]
            x[b, MID + 2 * i + 1] = vals[perm[i]]

    # Recall mask: query positions where target (next token) is the recalled value
    mask = torch.zeros(B, SEQ_LEN - 1, device=device)
    for i in range(PAIRS):
        mask[:, MID + 2 * i] = 1

    targets = x[:, 1:]
    return x, targets, mask


def train_and_eval(use_memory, device, steps=250, B=64, lr=1e-3):
    cfg = Config(
        vocab_size=VOCAB, d_model=128, d_state=16,
        d_conv=4, expand=2, n_layers=4, use_memory=use_memory,
    )
    model = HebbianMamba(cfg).to(device)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x, targets, mask = make_batch(B, device)

        logits1, _, mems = model(x[:, :MID])
        logits2, _, _ = model(x[:, MID:], memories=mems)
        logits = torch.cat([logits1, logits2], dim=1)[:, :-1]

        # Loss only on recall positions
        per_tok = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none")
        per_tok = per_tok.view(B, SEQ_LEN - 1)
        loss = (per_tok * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                preds = logits.argmax(-1)
                correct = ((preds == targets) * mask).sum()
                acc = (correct / mask.sum()).item()
            print(f"  step {step:4d} | loss {loss.item():.4f} | recall {acc:.1%}")

    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(20):
            x, targets, mask = make_batch(B, device)
            logits1, _, mems = model(x[:, :MID])
            logits2, _, _ = model(x[:, MID:], memories=mems)
            logits = torch.cat([logits1, logits2], dim=1)[:, :-1]
            preds = logits.argmax(-1)
            correct = ((preds == targets) * mask).sum()
            accs.append((correct / mask.sum()).item())
    return sum(accs) / len(accs)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: recall {PAIRS} pairs after Mamba reset")
    print(f"Chance: {1/NUM_VALS:.1%}\n")

    torch.manual_seed(42)
    print("With memory:")
    acc_mem = train_and_eval(use_memory=True, device=device)

    torch.manual_seed(42)
    print("\nWithout memory:")
    acc_nomem = train_and_eval(use_memory=False, device=device)

    print(f"\nRecall accuracy:")
    print(f"  With memory:    {acc_mem:.1%}")
    print(f"  Without memory: {acc_nomem:.1%}")
    print(f"  Chance:         {1/NUM_VALS:.1%}")


if __name__ == "__main__":
    main()
