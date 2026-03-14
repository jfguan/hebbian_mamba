"""Hebbian memory capacity benchmark.

Train on 64 KV pairs, then measure recall accuracy at [4, 8, 16, 32, 64, 128, 256].
Plots the capacity curve: where does the d=128 memory matrix start to fail?
"""

import sys

sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

from models.hebbian_components import CausalConv, GatedMLP, HebbianBlock

TRAIN_PAIRS = 64


class Model(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, d_conv=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Layer(d_model, d_conv) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.emb.weight = self.head.weight

    def forward(self, ids):
        x = self.emb(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class Layer(nn.Module):
    def __init__(self, D, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * D
        self.norm = nn.RMSNorm(D)
        self.mlp = GatedMLP(D, expand=expand)
        self.conv = CausalConv(d_inner, d_conv=d_conv)
        self.memory = HebbianBlock(D, memory_alpha=0.03, learned_alpha=False)

    def forward(self, x):
        normed = self.norm(x)
        out = self.mlp(normed, self.conv(self.mlp.project_up(normed)))
        out = self.memory(out)
        return x + out


def make_batch(B, pairs, device, n_keys=256, n_vals=16):
    # sequence: [k1 v1 k2 v2 ... | k3 v? k1 v? ...] where second half queries in shuffled order
    T = 4 * pairs
    mid = T // 2
    x = torch.zeros(B, T, dtype=torch.long, device=device)

    # random KV pairs (vals occupy [n_keys, n_keys+n_vals) so they don't collide with keys)
    keys = torch.stack(
        [torch.randperm(n_keys, device=device)[:pairs] for _ in range(B)]
    )
    vals = torch.randint(n_keys, n_keys + n_vals, (B, pairs), device=device)
    perms = torch.stack([torch.randperm(pairs, device=device) for _ in range(B)])

    # first half: store pairs, second half: query in shuffled order
    idx = torch.arange(pairs, device=device)
    x[:, 2 * idx] = keys
    x[:, 2 * idx + 1] = vals
    x[:, mid + 2 * idx] = keys.gather(1, perms)
    x[:, mid + 2 * idx + 1] = vals.gather(1, perms)

    # mask: only score value predictions in the query half
    mask = torch.zeros(B, T - 1, device=device)
    mask[:, mid + 2 * idx] = 1.0
    return x, x[:, 1:], mask


def plot_capacity(results, path, d_model, n_vals):
    matplotlib.use("Agg")

    # data
    pairs = list(results.keys())
    accs = [results[p] * 100 for p in pairs]
    guess_chance = 100 / n_vals

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pairs, accs, "o-", linewidth=2, markersize=8)
    ax.axhline(
        y=guess_chance,
        color="r",
        ls="--",
        alpha=0.5,
        label=f"chance ({guess_chance:.1f}%)",
    )

    # axes
    ax.set(
        xlabel="Number of KV pairs",
        ylabel="Recall accuracy (%)",
        title=f"Hebbian memory capacity (d_model={d_model})",
        ylim=(-5, 105),
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pairs)
    ax.set_xticklabels(pairs, rotation=90)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # save
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def train(model, device, vocab_size, steps=500):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        x, tgt, mask = make_batch(64, TRAIN_PAIRS, device)
        logits = model(x)[:, :-1]

        loss = (
            F.cross_entropy(
                logits.reshape(-1, vocab_size), tgt.reshape(-1), reduction="none"
            )
            .view_as(mask)
            .mul(mask)
            .sum()
            / mask.sum()
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                acc = ((logits.argmax(-1) == tgt) * mask).sum() / mask.sum()
            print(f"  step {step:3d}  loss={loss.item():.4f}  recall={acc.item():.1%}")


def eval_sweep(model, device):
    eval_pairs = [2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128]
    results = {}
    model.eval()
    print("capacity sweep:")
    with torch.no_grad():
        for p in eval_pairs:
            accs = []
            for _ in range(20):
                x, tgt, mask = make_batch(64, p, device)
                logits = model(x)[:, :-1]
                accs.append(((logits.argmax(-1) == tgt) * mask).sum() / mask.sum())
            results[p] = torch.stack(accs).mean().item()
            print(f"  {p:3d} pairs: {results[p]:.1%}")
    for p, acc in results.items():
        print(f"  {p},{acc:.4f}")
    return results


if __name__ == "__main__":
    n_vals = 16
    vocab_size = 256 + n_vals

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)

    model = Model(vocab_size=vocab_size, d_model=64, n_layers=4, d_conv=4).to(device)
    print(
        f"device={device}  params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  train_pairs={TRAIN_PAIRS}"
    )

    train(model, device, vocab_size)
    results = eval_sweep(model, device)
    plot_capacity(results, "eval_synthetic_recall/capacity.png", d_model=64, n_vals=n_vals)
