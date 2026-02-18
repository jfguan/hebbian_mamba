# 100M Scaling Experiment (Lambda GPU)

## Instance
- A10 (24GB VRAM) is sufficient. A100 if available.
- Estimated time: ~20-40 min total on A10 with compile + bf16

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
git clone https://github.com/jfguan/hebbian_mamba.git
cd hebbian_mamba
uv sync
```

## Train

```bash
# Memory model: d=1024, 12 layers, 105.7M params
uv run python train.py --steps 1000 --batch-size 4 --seq-len 2048 \
    --d-model 1024 --n-layers 12 --tag mem1_100M --compile --grad-accum 1

# Baseline: d=1024, 16 layers, 107.2M params (param-matched, no memory)
uv run python train.py --steps 1000 --batch-size 4 --seq-len 2048 \
    --d-model 1024 --n-layers 16 --no-memory --tag mem0_100M --compile --grad-accum 1
```

## Eval

```bash
# 32K context eval on both
uv run python eval_memory.py --model model_mem1_100M.pt \
    --tokens 32768 --windows 2 --segment 2048

uv run python eval_memory.py --model model_mem0_100M.pt \
    --tokens 32768 --windows 2 --segment 2048

# 64K if time permits
uv run python eval_memory.py --model model_mem1_100M.pt \
    --tokens 65536 --windows 2 --segment 4096

uv run python eval_memory.py --model model_mem0_100M.pt \
    --tokens 65536 --windows 2 --segment 4096
```

## Download results

```bash
# From your local machine
scp user@instance:hebbian_mamba/model_*100M.pt .
scp user@instance:hebbian_mamba/eval_memory.png .
scp user@instance:hebbian_mamba/history_*100M.jsonl .
scp user@instance:hebbian_mamba/loss_*100M.png .
```

## What we're looking for
- Memory model should beat baseline at same param count
- Gap should be larger than the 18M experiment (0.048 loss / 9% PPL at 32K)
- If gap grows → d² capacity scaling confirmed, strong paper result
- If gap shrinks or disappears → W doesn't survive scaling, interesting negative result
