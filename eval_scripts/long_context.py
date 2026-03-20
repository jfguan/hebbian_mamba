"""Long-context loss comparison across models.

Runs all models on the same val windows, measures per-segment loss.
Shows how each architecture handles increasing context depth.

Usage:
    uv run eval_scripts/long_context.py --models hebbian_100M mamba_100M gdn_100M --tokens 16384
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_dataset
from models import build_model

DATASET = "the_stack"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Checkpoint filenames in checkpoints/")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--segment", type=int, default=1024)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dataset = load_dataset(DATASET)
    print(f"device={device}  tokens={args.tokens}  windows={args.windows}  segments={args.tokens // args.segment}")

    # load models
    models = []
    for name in args.models:
        path = f"checkpoints/{name}.pt"
        model, label = load_model(path, device)
        models.append((model, name, label))

    # pick random val windows
    rng = np.random.default_rng(42)
    max_start = len(dataset.val) - args.tokens - 1
    starts = sorted(rng.choice(max_start, size=args.windows, replace=False))

    # eval
    n_segs = args.tokens // args.segment
    all_losses = np.zeros((len(models), args.windows, n_segs))

    for w_i, start in enumerate(starts):
        tokens = dataset.val[start : start + args.tokens + 1].tolist()
        print(f"\nwindow {w_i + 1}/{args.windows}: val[{start}:{start + args.tokens}]")

        for m_i, (model, name, label) in enumerate(models):
            losses = run_model(model, tokens, device)
            for s_i in range(n_segs):
                lo, hi = s_i * args.segment, (s_i + 1) * args.segment
                all_losses[m_i, w_i, s_i] = losses[lo:hi].mean()
            print(f"  {label}: {losses.mean():.4f}")

    # results
    mean_loss = all_losses.mean(axis=1)
    overall = all_losses.mean(axis=(1, 2))
    names = [name for _, name, _ in models]

    print_results(mean_loss, overall, names, args)
    plot_results(mean_loss, names, args)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model_config = ckpt["model_config"]
    model = build_model(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    label = f"{model_config.name} ({n_params:.1f}M)"
    return model, label


@torch.no_grad()
def run_model(model, tokens, device):
    losses = []
    states = None
    for t in range(len(tokens) - 1):
        tok = torch.tensor([tokens[t]], device=device)
        tgt = torch.tensor([tokens[t + 1]], device=device)
        logits, states = model.step(tok, states=states)
        states = detach(states)
        losses.append(F.cross_entropy(logits, tgt).item())
    return np.array(losses)


def detach(x):
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, dict):
        return {k: detach(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x)
    return x


def print_results(mean_loss, overall, names, args):
    seg_labels = [f"{s//1024}K-{(s + args.segment)//1024}K" for s in range(0, args.tokens, args.segment)]
    col_w = max(len(n) for n in names) + 2

    print(f"\n=== Per-segment loss (avg over {args.windows} windows) ===")
    print(f"{'Segment':>12}" + "".join(f"{n:>{col_w}}" for n in names))
    print("-" * (12 + col_w * len(names)))
    for s_i, lbl in enumerate(seg_labels):
        print(f"{lbl:>12}" + "".join(f"{mean_loss[m_i, s_i]:>{col_w}.4f}" for m_i in range(len(names))))
    print("-" * (12 + col_w * len(names)))
    print(f"{'OVERALL':>12}" + "".join(f"{overall[m_i]:>{col_w}.4f}" for m_i in range(len(names))))


def plot_results(mean_loss, names, args):
    seg_labels = [f"{s//1024}K-{(s + args.segment)//1024}K" for s in range(0, args.tokens, args.segment)]
    x = np.arange(len(seg_labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # per-segment loss
    for m_i, name in enumerate(names):
        ax1.plot(x, mean_loss[m_i], marker="o", markersize=4, label=name)
    ax1.set(ylabel="Loss (nats)", title=f"Per-Segment Loss ({args.tokens // 1024}K context)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # gap vs first model
    for m_i in range(1, len(names)):
        gap = mean_loss[m_i] - mean_loss[0]
        ax2.plot(x, gap, marker="o", markersize=4, label=f"{names[m_i]} − {names[0]}")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set(xlabel="Segment", ylabel=f"Loss gap vs {names[0]}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(seg_labels, rotation=45, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = f"eval_results/long_context_{args.tokens // 1024}k.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
