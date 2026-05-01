"""
Training Script for GPT Model

Reads all hyperparameters from config/default.yaml (or a custom config via --config).
Supports resume from checkpoint, gradient accumulation, mixed precision, and DDP.

Usage:
    python -m src.training.train
    python -m src.training.train --config config/default.yaml
    python -m src.training.train --set max_iters=50000 learning_rate=3e-4
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tokenizers import Tokenizer

from src.model.gpt import GPT, GPTConfig
from src.config import get_config_from_args


def train():
    # ─── Load Configuration ───────────────────────────────────────────────
    cfg = get_config_from_args()

    # Unpack config values
    out_dir = cfg["out_dir"]
    eval_interval = cfg["eval_interval"]
    log_interval = cfg["log_interval"]
    eval_iters = cfg["eval_iters"]
    eval_only = cfg["eval_only"]
    always_save_checkpoint = cfg["always_save_checkpoint"]
    init_from = cfg["init_from"]

    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    batch_size = cfg["batch_size"]
    block_size = cfg["block_size"]

    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_embd = cfg["n_embd"]
    dropout = cfg["dropout"]
    bias = cfg["bias"]

    learning_rate = cfg["learning_rate"]
    max_iters = cfg["max_iters"]
    weight_decay = cfg["weight_decay"]
    beta1 = cfg["beta1"]
    beta2 = cfg["beta2"]
    grad_clip = cfg["grad_clip"]

    decay_lr = cfg["decay_lr"]
    warmup_iters = cfg["warmup_iters"]
    lr_decay_iters = cfg["lr_decay_iters"]
    min_lr = cfg["min_lr"]

    backend = cfg["backend"]
    device = cfg["device"]
    dtype = cfg["dtype"]
    compile_model = cfg["compile"]

    data_dir = cfg["data_dir"]
    tokenizer_path = cfg["tokenizer_path"]

    # ─── Derived Values ───────────────────────────────────────────────────
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

    print("=" * 60)
    print("  GPT TRAINING")
    print("=" * 60)
    print(f"  Config loaded from: {cfg.get('_source', 'config/default.yaml')}")
    print(f"  Tokens per iteration: {tokens_per_iter:,}")
    print(f"  Model: {n_layer}L / {n_head}H / {n_embd}E / block_size={block_size}")
    print(f"  Training: {max_iters:,} iters, lr={learning_rate}, batch={batch_size}")
    print(f"  Device: {device}, dtype: {dtype}")
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    # ─── Data Loading ─────────────────────────────────────────────────────
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin = os.path.join(data_dir, 'val.bin')

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print(f"ERROR: Data files not found in {data_dir}/")
        print("Run: python -m src.data.prepare --dataset openwebtext")
        return

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')

    print(f"  Train data: {len(train_data):,} tokens ({os.path.getsize(train_bin) / 1e6:.1f} MB)")
    print(f"  Val data:   {len(val_data):,} tokens ({os.path.getsize(val_bin) / 1e6:.1f} MB)")

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # ─── Model Init ───────────────────────────────────────────────────────
    iter_num = 0
    best_val_loss = 1e9

    model_args = dict(
        n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
    )

    # Load tokenizer to get vocab size
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        model_args['vocab_size'] = tokenizer.get_vocab_size()
    else:
        print("WARNING: Tokenizer not found, using default vocab size 50304")
        model_args['vocab_size'] = 50304

    print(f"  Vocab size: {model_args['vocab_size']:,}")

    if init_from == 'scratch':
        print("\n  Initializing a new model from scratch...")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            print(f"\n  No checkpoint at {ckpt_path}, starting from scratch instead.")
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        else:
            print(f"\n  Resuming training from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = checkpoint_model_args[k]
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            # Fix keys from torch.compile / DDP
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            print(f"  Resumed at iter {iter_num}, best_val_loss={best_val_loss:.4f}")

    model.to(device)

    # ─── Optimizer ────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume' and os.path.exists(os.path.join(out_dir, 'ckpt.pt')):
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception:
            print("  WARNING: Could not load optimizer state, starting fresh optimizer")
        checkpoint = None  # free memory

    # Compile the model
    if compile_model:
        print("  Compiling model with torch.compile (takes ~1 minute)...")
        model = torch.compile(model)

    # ─── Loss Estimation ──────────────────────────────────────────────────
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # ─── LR Schedule ─────────────────────────────────────────────────────
    def get_lr(it):
        # 1) linear warmup
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) past decay → min lr
        if it > lr_decay_iters:
            return min_lr
        # 3) cosine decay
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # ─── Training Loop ────────────────────────────────────────────────────
    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    print(f"\n  Starting training loop at iter {iter_num}...\n")

    while True:
        # Set learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate and checkpoint
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"  step {iter_num:>6d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                if iter_num > 0:
                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': cfg,
                    }
                    print(f"    → saving checkpoint to {out_dir}")
                    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        # Forward / backward with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"  iter {iter_num:>6d}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint saved to: {out_dir}/ckpt.pt")


if __name__ == "__main__":
    train()
