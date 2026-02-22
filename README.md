# Hebbian Memory for Mamba

A Hebbian associative memory matrix $W \in \mathbb{R}^{d \times d}$ augmenting each Mamba layer. $W$ accumulates outer-product associations over the context at $O(Td^2)$ cost — linear in sequence length — and injects retrieved content additively into the residual stream. Drop-in: 32 lines of code, no training objective changes.

Paper: `paper/paper.tex`

## Results

| Setting | Memory | Baseline | Gap |
|---|---|---|---|
| 18M code (Python, 6M tokens) | 1.848 | 2.787 | **+0.939** |
| 18M prose (PG-19, 6M tokens) | 2.477 | 2.527 | **+0.050** |
| 100M code (Python, 16.4M tokens) | 1.661 | 1.968 | **+0.307** |
| 100M multilingual (The Stack, 8.2M tokens) | 1.621 | 1.884 | **+0.263** |
| 100M code @ 16K context | 1.685 | 2.109 | **+0.424** |
| 100M Stack @ 16K context | 1.453 | 2.092 | **+0.639** |

The 18M memory model outperforms the 100M parameter-matched baseline throughout its entire training run (1.848 vs 1.968 best checkpoint), at 5.6× fewer parameters.

## Replication

### Setup

```bash
uv sync
```

Data is cached automatically on first run from HuggingFace.

### Synthetic recall sanity check

Verifies that $W$ achieves near-perfect associative recall before training:

```bash
uv run eval_recall/eval_recall.py
```

### 18M prose (PG-19)

```bash
# Memory model
uv run train.py --dataset pg19

# Wide baseline (~17.4M, d=576)
uv run train.py --dataset pg19 --no-memory --d-model 576

# Deep baseline (~17.2M, 10 layers)
uv run train.py --dataset pg19 --no-memory --n-layers 10
```

### 18M code (codeparrot)

```bash
uv run train.py --dataset code
uv run train.py --dataset code --no-memory --d-model 576
uv run train.py --dataset code --no-memory --n-layers 10
```

### 100M code (codeparrot)

```bash
uv run train_100M.py --dataset code
uv run train_100M.py --dataset code --no-memory
```

### 100M multilingual (The Stack)

```bash
uv run train_stack100M.py
uv run train_stack100M.py --no-memory
```

### Evaluation

**W ablation** (updating vs frozen $W$, confirms gains come from $W$ not extra params):

```bash
uv run eval_memory/eval_memory.py --model checkpoints/model_code_memory.pt --dataset code --tokens 16384 --windows 8
```

**Long-context eval** (per-segment loss up to 16K tokens):

```bash
# Run from repo root
uv run eval_long/eval_long.py \
  --models checkpoints/model_code_memory.pt checkpoints/model_code_deep.pt \
  --dataset code --tokens 16384 --windows 8 \
  --out eval_long/my_eval.txt
```

**Compile paper:**

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex \
  -output-directory=paper paper/paper.tex
```

## Architecture

Each Mamba layer is augmented with:

```
W_t = γ · W_{t-1} + proj_write(r_{t-1}) ⊗ proj_write(r_{t-1})ᵀ   # write
read_t = W_t · proj_read(r_t)                                        # read
r_t ← r_t + α · proj_read(read_t)                                   # inject
```

where `r_t` is the post-Mamba residual, `γ = σ(decay)` is a learned per-layer scalar, and `α = 0.03` is fixed. Two key design choices: (1) separate write/read projections — necessary for correct retrieval after Mamba state resets; (2) fixed `α` — prevents memory noise from destabilizing early training.

## Checkpoints

| File | Params | Dataset |
|---|---|---|
| `checkpoints/model_mem1.pt` | 18M | PG-19 prose |
| `checkpoints/model_code_memory.pt` | 18M | codeparrot |
| `checkpoints/model_code_deep.pt` | 17.2M | codeparrot (baseline) |
| `checkpoints/model_code100M_memory.pt` | 105.7M | codeparrot |
| `checkpoints/model_code100M_deep.pt` | 107.2M | codeparrot (baseline) |
| `checkpoints/model_stack100M_memory.pt` | 97.5M | The Stack |
| `checkpoints/model_stack100M_deep.pt` | 101.1M | The Stack (baseline) |
