"""Generate code training curves for paper — two subplots: 18M and 100M."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(tag, tps):
    entries = [json.loads(l) for l in open(f"checkpoints/history_{tag}.jsonl") if "val_loss" in l]
    tokens = [e["tokens"] / 1e6 if e.get("tokens", 0) > 0 else e["step"] * tps / 1e6
              for e in entries]
    vals = [e["val_loss"] for e in entries]
    return tokens, vals


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# --- 18M models (B=2, T=2048 → 4096 tokens/step) ---
tps_18m = 4096
for tag, label, color, ls in [
    ("code_memory", "Memory (d=512, 8L)",        "#1f77b4", "-"),
    ("code_deep",   "Deep baseline (d=512, 10L)", "#2ca02c", "--"),
]:
    t, v = load(tag, tps_18m)
    ax1.plot(t, v, ls, color=color, label=label, linewidth=1.8, marker="o", markersize=3)
ax1.set_xlabel("Training tokens (M)")
ax1.set_ylabel("Val loss (nats)")
ax1.set_title("18M parameters")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- 100M models (B=1, T=2048 → 2048 tokens/step) ---
tps_100m = 2048
for tag, label, color, ls in [
    ("code100M_memory", "Memory (d=1024, 12L)",        "#1f77b4", "-"),
    ("code100M_deep",   "Deep baseline (d=1024, 16L)", "#2ca02c", "--"),
]:
    t, v = load(tag, tps_100m)
    ax2.plot(t, v, ls, color=color, label=label, linewidth=1.8, marker="o", markersize=3)
ax2.set_xlabel("Training tokens (M)")
ax2.set_title("100M parameters")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("Python code validation loss — memory vs param-matched baseline", fontsize=11)
fig.tight_layout()
fig.savefig("figures/code_training.pdf", dpi=150)
fig.savefig("figures/code_training.png", dpi=150)
print("Saved figures/code_training.pdf and .png")
