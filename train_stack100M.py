"""Train a ~100M param HebbianMamba on The Stack (multilingual code, 1024-vocab BPE).

Architecture: d_model=1024, n_layers=11 memory / 15 deep (~97M / ~101M params).
Vocab size 1024 (vs 512 for code dataset) absorbs ~0.5M params in embedding/lm_head.
LR schedule: cosine over 64K steps (~131M tokens at B=1, T=2048).
Resume with --resume; run in chunks with --steps.
"""

import argparse
import json
import os
import time

import torch

from model import Config, HebbianMamba
from train import cosine_lr, configure_optimizers, evaluate, sample, plot_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=64_000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--memory-alpha", type=float, default=0.04)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--ckpt-interval", type=int, default=2000)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    from data_stack import load_dataset, DataLoader

    use_memory = not args.no_memory
    tag = args.tag or ("stack100M_memory" if use_memory else "stack100M_deep")

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    device_type = device.split(":")[0]
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = load_dataset()
    train_loader = DataLoader(ds["train"], args.batch_size, args.seq_len)
    val_loader = DataLoader(ds["val"], args.batch_size, args.seq_len)

    n_layers = 11 if use_memory else 15
    cfg = Config(
        vocab_size=ds["vocab_size"],
        d_model=1024,
        n_layers=n_layers,
        d_state=16,
        use_memory=use_memory,
        memory_alpha=args.memory_alpha,
    )
    model = HebbianMamba(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = configure_optimizers(model, args.lr, weight_decay=0.1)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New params: {missing}")
        if not missing and not unexpected and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    end_step = min(start_step + args.steps, args.total_steps)
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    min_lr = args.lr * 0.1

    print(f"{n_params/1e6:.1f}M params | memory={use_memory} | alpha={args.memory_alpha} | {device}")
    print(f"Steps {start_step} -> {end_step} / {args.total_steps} | B={args.batch_size}x{args.grad_accum} T={args.seq_len} lr={args.lr}")
    print(f"Tokens/step: {tokens_per_step:,} | ~{tokens_per_step * (end_step - start_step) / 1e6:.1f}M tokens this run")
    print(f"Full schedule: {args.total_steps * tokens_per_step / 1e6:.0f}M tokens total")

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"checkpoints/history_{tag}.jsonl"
    log_file = open(log_path, "a" if args.resume else "w")

    step = start_step
    entry = {}
    tokens_seen = start_step * tokens_per_step
    try:
        for step in range(start_step, end_step):
            t0 = time.time()
            lr = cosine_lr(step, args.warmup, args.total_steps, args.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for _ in range(args.grad_accum):
                x, y = train_loader.batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / args.grad_accum
                loss.backward()
                loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tokens_seen = (step + 1) * tokens_per_step
            dt = time.time() - t0
            entry = {"step": step, "train_loss": loss_accum, "tokens": tokens_seen}
            print(
                f"step {step:5d} | loss {loss_accum:.4f} | tokens {tokens_seen/1e6:.1f}M"
                f" | lr {lr:.2e} | {dt*1000:.0f}ms",
                flush=True,
            )

            if step > 0 and step % args.eval_interval == 0:
                vl = evaluate(model, val_loader, device)
                entry["val_loss"] = vl
                print(f"  val loss {vl:.4f} | val ppl {__import__('math').exp(vl):.2f}", flush=True)

            if step > 0 and step % args.ckpt_interval == 0:
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                ckpt_path = f"checkpoints/ckpt_{tag}_step{step}.pt"
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": cfg, "step": step}, ckpt_path)
                print(f"  -> {ckpt_path}", flush=True)

            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

    except KeyboardInterrupt:
        print(f"\nStopped at step {step}.")

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    vl = evaluate(model, val_loader, device)
    log_file.write(json.dumps({"step": step, "train_loss": entry.get("train_loss", 0),
                               "val_loss": vl, "tokens": tokens_seen}) + "\n")
    log_file.close()
    print(f"\nFinal val loss: {vl:.4f} | ppl {__import__('math').exp(vl):.2f}")

    sample_prompt = 'def fizzbuzz(n):\n    """Print 1 to n; Fizz for multiples of 3, Buzz for 5, FizzBuzz for both."""\n'
    print(f"Sample:\n{sample(raw_model, ds['encode'], ds['decode'], device, prompt=sample_prompt, n=300)}")

    ckpt_path = f"checkpoints/model_{tag}.pt"
    torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg, "step": step + 1}, ckpt_path)
    history = [json.loads(line) for line in open(log_path)]
    plot_losses(history, tag)
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
