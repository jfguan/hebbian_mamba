# Cloud Training (Lambda A100)

## 1. SSH Key Setup

```bash
# Generate key (if you don't have one)
ssh-keygen -t ed25519 -C "jeff.guan0@gmail.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub
# Paste into Lambda dashboard: https://cloud.lambdalabs.com/ssh-keys

# SSH into instance
ssh ubuntu@<instance-ip>
```

## 2. Helix Editor

```bash
# Install helix
sudo add-apt-repository ppa:maveonair/helix-editor
sudo apt update && sudo apt install helix

# Or via snap
sudo snap install helix --classic

# Usage
hx <file>
```

## 3. Repo Setup

```bash
# Clone repo
git clone git@github.com:<user>/hebbian_mamba.git
cd hebbian_mamba

# Install deps
pip install torch numpy matplotlib huggingface_hub
```

## 4. Download Data (~20GB)

```bash
python bench_fineweb/data.py
```

## 5. Train

```bash
python bench_fineweb/train_cloud.py
```

Default: batch_size=32, grad_accum=16, 524K tokens/step, ~10B tokens total.

If OOM, halve batch_size and double grad_accum:
```bash
python bench_fineweb/train_cloud.py --batch-size 16 --grad-accum 32
python bench_fineweb/train_cloud.py --batch-size 8 --grad-accum 64
python bench_fineweb/train_cloud.py --batch-size 4 --grad-accum 128
```

## 6. Resume (if interrupted)

```bash
python bench_fineweb/train_cloud.py --resume bench_fineweb/checkpoints/ckpt_cloud_step2000.pt
```

## 7. Copy Results Back

```bash
# From local machine
scp ubuntu@<instance-ip>:~/hebbian_mamba/bench_fineweb/checkpoints/history_cloud.jsonl bench_fineweb/checkpoints/
scp ubuntu@<instance-ip>:~/hebbian_mamba/bench_fineweb/checkpoints/model_cloud.pt bench_fineweb/checkpoints/
```
