"""Generate Stack 100M long-context per-segment chart for paper."""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "eval_long/16384tok_8win_stack100M_1k.txt"

memory_vals, deep_vals = [], []
for line in open(LOG):
    if re.match(r"\s+\d+K--\d+K", line):
        parts = line.split()
        memory_vals.append(float(parts[1]))
        deep_vals.append(float(parts[2]))

assert len(memory_vals) == 16, f"Expected 16 segments, got {len(memory_vals)}"

x_k = [i + 0.5 for i in range(16)]
gap = [d - m for m, d in zip(memory_vals, deep_vals)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(x_k, memory_vals, "-", color="#1f77b4", label="Memory (d=1024, 11L)",
         linewidth=1.8, marker="o", markersize=4)
ax1.plot(x_k, deep_vals, "--", color="#2ca02c", label="Deep baseline (d=1024, 15L)",
         linewidth=1.8, marker="s", markersize=4)
ax1.set_ylabel("Val loss (nats)")
ax1.set_title("Stack 100M long-context evaluation — 8 × 16K multilingual code windows")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.axvline(2.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)

ax2.fill_between(x_k, gap, alpha=0.3, color="#d62728")
ax2.plot(x_k, gap, "-", color="#d62728", linewidth=1.5)
ax2.axhline(0, color="gray", linewidth=0.8)
ax2.axvline(2.0, color="gray", linestyle=":", linewidth=1, alpha=0.6,
            label="Train length (2K)")
ax2.set_xlabel("Context position (K tokens)")
ax2.set_ylabel("Gap (nats)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("figures/long_context_stack.pdf", dpi=150)
fig.savefig("figures/long_context_stack.png", dpi=150)
print("Saved figures/long_context_stack.pdf and .png")
