"""Hebbian memory capacity benchmark.

Train on KV pairs, then measure recall accuracy at increasing pair counts.
Plots the capacity curve: where does the memory matrix start to fail?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.hebbian_components import CausalConv, GatedMLP, HebbianBlock

# -- config --
N_KEYS = 256
N_VALS = 16
VOCAB = N_KEYS + N_VALS
TRAIN_PAIRS = 32


def main():
    torch.manual_seed(42)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = Model(d_model=128, n_layers=4).to(device)
    param_sum = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={device}  params={param_sum:.2f}M  train_pairs={TRAIN_PAIRS}")

    train(model, device)
    results = sweep(model, device)
    plot_capacity(results, "eval_results/synthetic_recall.png", d_model=128)


def make_batch(batch_size, num_pairs, device):
    """Generate a KV recall task.

    Store phase: model sees [k1, v1, k2, v2, ...] — memorize these pairs.
    Query phase: model sees [k?, _, k?, _, ...] — predict the value for each key.
    Keys are reshuffled in query phase so the model can't just memorize order.

    returns: (input_ids, targets, mask) where mask marks value prediction positions.
    """
    B, P = batch_size, num_pairs

    # generate random KV pairs
    keys = torch.stack([torch.randperm(N_KEYS, device=device)[:P] for _ in range(B)])
    values = torch.randint(N_KEYS, VOCAB, (B, P), device=device)

    # store phase: interleave keys and values [k1, v1, k2, v2, ...]
    store_seq = torch.stack([keys, values], dim=-1).view(B, 2 * P)

    # query phase: same pairs but shuffled order
    shuffle = torch.stack([torch.randperm(P, device=device) for _ in range(B)])
    query_seq = torch.stack([
        keys.gather(1, shuffle),
        values.gather(1, shuffle),
    ], dim=-1).view(B, 2 * P)

    # full sequence: [store | query]
    input_ids = torch.cat([store_seq, query_seq], dim=1)

    # mask: only score value predictions in query phase
    targets = input_ids[:, 1:]
    mask = torch.zeros_like(targets, dtype=torch.float)
    mask[:, 2 * P :: 2] = 1.0
    return input_ids, targets, mask


def train(model, device, steps=1500):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        # generate a batch of KV recall tasks
        x, targets, mask = make_batch(64, TRAIN_PAIRS, device)

        # predict next token, compute loss only on recall positions
        logits = model(x)[:, :-1]
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none"
        )
        masked_loss = per_token_loss.view_as(mask) * mask
        loss = masked_loss.sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log recall accuracy at recall positions only
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                predictions = logits.argmax(-1)
                correct = (predictions == targets) * mask
                accuracy = correct.sum() / mask.sum()
            print(f"step {step:3d}  loss={loss.item():.4f}  recall={accuracy.item():.1%}")


def sweep(model, device):
    """Test recall accuracy at increasing numbers of KV pairs."""
    print("capacity sweep:")
    model.eval()

    results = {}
    with torch.no_grad():
        for num_pairs in [4, 8, 16, 32, 64, 96, 128, 160, 192, 256]:
            accuracies = []
            for _ in range(20):
                x, targets, mask = make_batch(64, num_pairs, device)
                predictions = model(x)[:, :-1].argmax(-1)
                correct = (predictions == targets) * mask
                accuracies.append(correct.sum() / mask.sum())

            results[num_pairs] = torch.stack(accuracies).mean().item()
            print(f"  {num_pairs:3d} pairs: {results[num_pairs]:.1%}")

    return results


def plot_capacity(results, path, d_model):
    num_pairs = list(results.keys())
    accuracy_pct = [results[p] * 100 for p in num_pairs]
    guess_chance = 100 / N_VALS

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(num_pairs, accuracy_pct, "o-", linewidth=2, markersize=8)
    ax.axhline(y=guess_chance, color="r", ls="--", alpha=0.5, label=f"chance ({guess_chance:.1f}%)")
    ax.set(
        xlabel="Number of KV pairs",
        ylabel="Recall accuracy (%)",
        title=f"Hebbian memory capacity (d_model={d_model})",
        ylim=(-5, 105),
    )
    ax.set_xticks(num_pairs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


# -- model --

class Model(nn.Module):
    def __init__(self, d_model, n_layers, d_conv=4):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, d_model)
        self.layers = nn.ModuleList([Layer(d_model, d_conv=d_conv) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)
        self.embedding.weight = self.head.weight

    def forward(self, ids):
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class Layer(nn.Module):
    def __init__(self, d_model, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.norm = nn.RMSNorm(d_model)
        self.mlp = GatedMLP(d_model, expand=expand)
        self.conv = CausalConv(d_inner, d_conv=d_conv)
        self.memory = HebbianBlock(d_model)

    def forward(self, x):
        normed = self.norm(x)
        out = self.mlp(normed, self.conv(self.mlp.project_up(normed)))
        out = self.memory(out)
        return x + out


if __name__ == "__main__":
    main()
