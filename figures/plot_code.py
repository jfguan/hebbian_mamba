"""Generate code training curve for paper (figures/code_training.pdf)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TPS = 4096  # tokens per step: B=2, T=2048


def load(tag):
    entries = [json.loads(l) for l in open(f"checkpoints/history_{tag}.jsonl") if "val_loss" in l]
    tokens = [e.get("tokens", e["step"] * TPS) / 1e6 for e in entries]
    vals = [e["val_loss"] for e in entries]
    return tokens, vals


fig, ax = plt.subplots(figsize=(7, 4))
styles = [
    ("code_memory", "Memory (d=512, 8L)",        "#1f77b4", "-"),
    ("code_deep",   "Deep baseline (d=512, 10L)", "#2ca02c", "--"),
]
for tag, label, color, ls in styles:
    t, v = load(tag)
    ax.plot(t, v, ls, color=color, label=label, linewidth=1.8, marker="o", markersize=3)

ax.set_xlabel("Training tokens (M)")
ax.set_ylabel("Val loss (nats)")
ax.set_title("Python code validation loss — memory vs param-matched baseline")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("figures/code_training.pdf", dpi=150)
fig.savefig("figures/code_training.png", dpi=150)
print("Saved figures/code_training.pdf and .png")
