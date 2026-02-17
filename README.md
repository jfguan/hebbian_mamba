# Continual Learning via Persistent Memory in Mamba

## Core Idea

Add a slow-timescale memory matrix W ∈ ℝ^{d×d} alongside Mamba's fast recurrent state at each layer. Train with periodic resets of Mamba's state (never W) to force long-range knowledge through the persistent memory. At inference, the model improves on a domain simply by reading more of it — no gradient updates required.

## Why Mamba, Not Transformers

Transformers have lossless KV-cache memory within the context window. A lossy memory matrix cannot compete with exact attention and gets ignored during training. Mamba's state is already lossy, so a complementary slow memory fills a real gap.

## Update Rules (per layer)

Let r_t be the residual-stream vector after Mamba's recurrent update.

**Read:** $\text{read}_t = W \cdot r_{t-1}$

**Write:** $W \leftarrow \sigma(\lambda) \cdot W + \text{proj}_{\text{write}}(r_t) \cdot r_{t-1}^\top$

**Inject:** $r_t \leftarrow r_t + \alpha \cdot \text{proj}_{\text{read}}(\text{read}_t)$

Learned parameters per layer: one scalar λ (decay), two projections, one fixed scalar α = 0.03. The write rule is deliberately simple — one matrix, one decay, one outer product. Mamba's selectivity mechanism pre-filters local noise, so the write signal is already clean.

## Training: Multi-Reset Chunking

During training, each 2048-token sequence is split into 256-token chunks. Mamba's hidden state resets at each chunk boundary. W is never reset — it carries across all chunks via detached memory passing (BPTT truncated per chunk).

This provides: (1) clean gradient signal to W — after a reset, any useful long-range information must come from W; (2) forced timescale separation without architectural gating or curriculum design; (3) 8 resets per sequence, giving W repeated practice at bridging gaps.

At inference, no resets occur. Both systems run continuously.

## Architecture

- 8-layer Mamba (d_model=512, d_state=16, expand=2)
- Hebbian memory W ∈ ℝ^{512×512} at each layer
- BPE tokenizer (vocab_size=512) trained on PG-19 novels
- ~18M parameters total

## Dataset

PG-19 (Project Gutenberg novels) streamed via HuggingFace. ~10M chars train, ~1M chars val. Tokenized and cached as `.npy` files.

## Results

### W Updating vs W Frozen (within-model test)

Same model, two eval passes over 4096 val tokens (4 random windows, averaged):

| | Avg Loss | PPL |
|---|---|---|
| **W updating** | **3.083** | **~21.8** |
| W frozen | 3.176 | ~23.9 |

**Overall delta: +0.092 ± 0.010** (9.6 standard errors from zero).

Per-segment breakdown (512-token segments):

| Segment | Delta | ±stderr |
|---------|-------|---------|
| 0-512 | +0.073 | 0.021 |
| 512-1024 | +0.103 | 0.025 |
| 1024-1536 | +0.104 | 0.025 |
| 1536-2048 | +0.113 | 0.036 |
| 2048-2560 | +0.068 | 0.020 |
| 2560-3072 | +0.083 | 0.025 |
| 3072-3584 | +0.097 | 0.028 |
| 3584-4096 | +0.098 | 0.025 |

W helps at every position, including beyond the 2048-token training length.

### Memory vs Param-Matched Baseline (cross-model test)

Baseline: no memory, d_model=576 (~17.4M params, slightly fewer than memory model's 18M). Both trained with identical 256-token chunk resets for 1000 steps. Evaluated on same 4 random windows of 4096 tokens, no resets.

| Model | Avg Loss | PPL |
|---|---|---|
| **Memory (W updating)** | **3.083** | **~21.8** |
| Baseline (no memory, d_model=576) | 3.123 | ~22.7 |
| Memory (W frozen) | 3.176 | ~23.9 |

The memory model with W active beats the param-matched baseline by 0.040 loss (~4% perplexity). The baseline has a wider Mamba but no mechanism for cross-chunk information transfer. W provides something a bigger Mamba cannot.

### Learned Decay Rates

All layers learned decay σ(λ) ≈ 0.986–0.990, meaning W retains ~99% of its content per step. At this rate, information persists for hundreds of tokens — well beyond the 256-token chunk boundaries.

| Layer | σ(decay) |
|-------|----------|
| 0 | 0.989 |
| 1 | 0.990 |
| 2 | 0.988 |
| 3 | 0.988 |
| 4 | 0.987 |
| 5 | 0.987 |
| 6 | 0.987 |
| 7 | 0.985 |

## Evaluation

**2×2 factorial** (resets × persistent memory):

|  | No resets | Resets |
|---|---|---|
| **No memory** | `--no-memory --no-resets --d-model 576` | `--no-memory --d-model 576` |
| **Memory** | `--no-resets` | (default) |

## Scaling Outlook

W's advantage should grow along two axes:

- **Sequence length**: Mamba's state decays; W persists indefinitely. Longer sequences = bigger gap. Already validated: benefit holds at 4096 tokens (2× training length).
- **Model width**: W is d×d, so capacity scales as d². Larger models = more association storage before interference.

## Risks

1. **Interference.** Rank-1 updates with a single scalar decay may cause catastrophic forgetting as W accumulates. Fallback: discrete memory slots M ∈ ℝ^{k×d} with per-slot decay rates.
2. **Subsidization.** W may absorb work Mamba could handle alone, weakening the base model. The factorial design measures this directly.
3. **O(d²) per token per layer.** Manageable with low-rank projections or applying memory at every nth layer.

## Usage

```bash
# Train with memory + resets (default)
uv run python train.py --steps 1000 --batch-size 1

# Train param-matched baseline
uv run python train.py --steps 1000 --batch-size 1 --no-memory --d-model 576

# Evaluate (4 random windows, 4096 tokens each)
uv run python eval_memory.py --model model_mem1_reset1.pt

# Curriculum: pretrain without memory, then introduce W
uv run python train.py --steps 500 --batch-size 1 --no-memory --no-resets
uv run python train.py --steps 500 --batch-size 1 --resume model_mem0_reset0.pt
```
