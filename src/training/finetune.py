"""
Fine-tuning Script — Instruction Tuning for GPT

Takes a pretrained checkpoint and fine-tunes it on instruction-following data
(e.g., Alpaca, Dolly, OpenAssistant) to make the model useful for chat / Q&A.

Supports:
  - Alpaca-style: {"instruction", "input", "output"}
  - ShareGPT-style: {"conversations": [{"from": "human", "value": ...}, ...]}
  - Custom: any dataset with a "text" field

Usage:
    python -m src.training.finetune
    python -m src.training.finetune --dataset tatsu-lab/alpaca
    python -m src.training.finetune --config config/default.yaml --set finetune.max_iters=3000
"""

import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from tokenizers import Tokenizer

from src.model.gpt import GPT, GPTConfig
from src.config import get_config_from_args


# ──────────────────────────────────────────────────────────────────────────────
# Data formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_alpaca(example, tokenizer):
    """Format Alpaca-style instruction data with special tokens."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    
    if inp:
        prompt = f"<|user|>\n{instruction}\n{inp}\n<|assistant|>\n{output}<|endoftext|>"
    else:
        prompt = f"<|user|>\n{instruction}\n<|assistant|>\n{output}<|endoftext|>"
    
    return prompt


def format_sharegpt(example, tokenizer):
    """Format ShareGPT-style conversation data."""
    conversations = example.get("conversations", [])
    parts = []
    for turn in conversations:
        role = turn.get("from", "human")
        value = turn.get("value", "")
        if role == "human":
            parts.append(f"<|user|>\n{value}")
        elif role == "gpt":
            parts.append(f"<|assistant|>\n{value}")
    
    text = "\n".join(parts) + "<|endoftext|>"
    return text


def format_text(example, tokenizer):
    """Simple text-only format."""
    return example.get("text", "") + "<|endoftext|>"


FORMATTERS = {
    "alpaca": format_alpaca,
    "sharegpt": format_sharegpt,
    "text": format_text,
}


def detect_format(dataset):
    """Auto-detect dataset format from column names."""
    cols = dataset.column_names
    if "instruction" in cols and "output" in cols:
        return "alpaca"
    elif "conversations" in cols:
        return "sharegpt"
    else:
        return "text"


# ──────────────────────────────────────────────────────────────────────────────
# Prepare fine-tuning data
# ──────────────────────────────────────────────────────────────────────────────

def prepare_finetune_data(
    dataset_name,
    tokenizer,
    block_size,
    output_dir,
    split="train",
    val_ratio=0.05,
    max_examples=None,
):
    """
    Download, format, tokenize, and save fine-tuning data.
    
    Returns paths to train.bin and val.bin
    """
    from datasets import load_dataset
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "finetune_train.bin")
    val_path = os.path.join(output_dir, "finetune_val.bin")
    
    # Check if already prepared
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"  Fine-tune data already exists in {output_dir}, reusing.")
        return train_path, val_path
    
    print(f"  Loading fine-tune dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    # Detect format
    fmt = detect_format(dataset)
    formatter = FORMATTERS[fmt]
    print(f"  Detected format: {fmt}")
    print(f"  Examples: {len(dataset):,}")
    
    # Format all examples
    all_ids = []
    for example in dataset:
        text = formatter(example, tokenizer)
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
    
    print(f"  Total tokens: {len(all_ids):,}")
    
    # Split
    n = len(all_ids)
    split_idx = int(n * (1.0 - val_ratio))
    train_ids = np.array(all_ids[:split_idx], dtype=np.uint16)
    val_ids = np.array(all_ids[split_idx:], dtype=np.uint16)
    
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    
    print(f"  Train: {len(train_ids):,} tokens → {train_path}")
    print(f"  Val:   {len(val_ids):,} tokens → {val_path}")
    
    return train_path, val_path


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ──────────────────────────────────────────────────────────────────────────────

def finetune():
    """Main fine-tuning function."""
    cfg = get_config_from_args()
    ft_cfg = cfg.get("finetune", {})
    
    # Fine-tuning specific config
    base_checkpoint = ft_cfg.get("base_checkpoint", "out/ckpt.pt")
    out_dir = ft_cfg.get("output_dir", "out/finetuned")
    dataset_name = ft_cfg.get("dataset", "tatsu-lab/alpaca")
    dataset_split = ft_cfg.get("dataset_split", "train")
    max_iters = ft_cfg.get("max_iters", 5000)
    learning_rate = ft_cfg.get("learning_rate", 2e-5)
    dropout_rate = ft_cfg.get("dropout", 0.1)
    warmup_iters = ft_cfg.get("warmup_iters", 100)
    eval_interval = ft_cfg.get("eval_interval", 50)
    eval_iters = ft_cfg.get("eval_iters", 20)
    batch_size = ft_cfg.get("batch_size", 8)
    gradient_accumulation_steps = ft_cfg.get("gradient_accumulation_steps", 4)
    grad_clip = ft_cfg.get("grad_clip", 1.0)
    
    device = cfg["device"]
    dtype = cfg["dtype"]
    tokenizer_path = cfg["tokenizer_path"]
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
    
    print("=" * 60)
    print("  GPT FINE-TUNING (Instruction Tuning)")
    print("=" * 60)
    print(f"  Base checkpoint: {base_checkpoint}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Max iters: {max_iters}, LR: {learning_rate}")
    print(f"  Dropout: {dropout_rate}")
    print(f"  Device: {device}, dtype: {dtype}")
    print("=" * 60)
    
    # ─── Load base model ──────────────────────────────────────────────────
    if not os.path.exists(base_checkpoint):
        print(f"ERROR: Base checkpoint not found: {base_checkpoint}")
        print("Please train the base model first: python -m src.training.train")
        return
    
    print(f"\n  Loading base model from {base_checkpoint}...")
    checkpoint = torch.load(base_checkpoint, map_location=device)
    model_args = checkpoint['model_args']
    
    # Override dropout for fine-tuning
    model_args['dropout'] = dropout_rate
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    
    block_size = model_args['block_size']
    
    print(f"  Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")
    
    # ─── Load tokenizer ──────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # ─── Prepare fine-tuning data ─────────────────────────────────────────
    data_dir = os.path.join("data", "finetuning")
    train_path, val_path = prepare_finetune_data(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        block_size=block_size,
        output_dir=data_dir,
        split=dataset_split,
    )
    
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
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
    
    # ─── Optimizer (lower LR for fine-tuning) ─────────────────────────────
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )
    
    # ─── Loss estimation ─────────────────────────────────────────────────
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # ─── LR schedule ─────────────────────────────────────────────────────
    min_lr = learning_rate * 0.1
    
    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        if it > max_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
    
    # ─── Training loop ───────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = 1e9
    
    X, Y = get_batch('train')
    t0 = time.time()
    
    print(f"\n  Starting fine-tuning...\n")
    
    for iter_num in range(max_iters + 1):
        # Set LR
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            print(f"  step {iter_num:>5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                ckpt = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': ft_cfg,
                }
                print(f"    → New best! Saving to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
        
        # Forward / backward
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()
        
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % 10 == 0 and iter_num > 0:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"  iter {iter_num:>5d}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    
    print(f"\n{'='*60}")
    print(f"  FINE-TUNING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {out_dir}/ckpt.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    finetune()
