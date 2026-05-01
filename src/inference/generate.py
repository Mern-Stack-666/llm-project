"""
Text Generation — Generate text from a prompt using a trained checkpoint.

Usage:
    python -m src.inference.generate
    python -m src.inference.generate "The history of computing"
    python -m src.inference.generate --checkpoint out/finetuned/ckpt.pt --prompt "Explain AI"
"""

import os
import sys
import argparse
import torch
from tokenizers import Tokenizer
from src.model.gpt import GPT, GPTConfig


def generate_text(
    prompt,
    checkpoint_path="out/ckpt.pt",
    tokenizer_path="data/tokenizer.json",
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    device="auto",
):
    """Generate text from a prompt using a trained model."""
    
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path} on {device}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        print(f"  Model: {model.get_num_params() / 1e6:.2f}M params")
        print(f"  Trained for {checkpoint.get('iter_num', '?')} iters, val loss {checkpoint.get('best_val_loss', '?')}")
    else:
        print(f"No checkpoint at {checkpoint_path}, using random model (for testing)")
        gptconf = GPTConfig(
            n_layer=6, n_head=6, n_embd=384, block_size=256,
            vocab_size=tokenizer.get_vocab_size(), dropout=0.0
        )
        model = GPT(gptconf)

    model.eval()
    model.to(device)
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    print(f"\nPrompt: '{prompt}'")
    print("-" * 60)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print("-" * 60)
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from GPT model")
    parser.add_argument("prompt", nargs="?", default="Artificial intelligence is",
                        help="Text prompt to generate from")
    parser.add_argument("--checkpoint", type=str, default="out/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    generate_text(
        prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
