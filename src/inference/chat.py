"""
Interactive Chat — Chat with your trained GPT model.

Usage:
    python -m src.inference.chat
    python -m src.inference.chat --checkpoint out/finetuned/ckpt.pt
"""

import os
import argparse
import torch
from tokenizers import Tokenizer
from src.model.gpt import GPT, GPTConfig


def chat(
    checkpoint_path="out/ckpt.pt",
    tokenizer_path="data/tokenizer.json",
    max_new_tokens=150,
    temperature=0.8,
    top_k=40,
    device="auto",
):
    """Interactive chat loop with the GPT model."""
    
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load model
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Please train the model first.")
        return
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    block_size = checkpoint['model_args'].get('block_size', 256)
    
    print(f"\n{'='*50}")
    print(f"  GPT CHAT ({model.get_num_params() / 1e6:.1f}M params on {device})")
    print(f"  Type 'exit' to quit, 'clear' to reset history")
    print(f"{'='*50}")
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if user_input.lower() == 'clear':
            conversation_history = []
            print("  [History cleared]")
            continue
        
        # Build prompt with conversation history
        conversation_history.append(("user", user_input))
        
        # Format as chat prompt (works best with fine-tuned model)
        parts = []
        for role, content in conversation_history:
            if role == "user":
                parts.append(f"<|user|>\n{content}")
            else:
                parts.append(f"<|assistant|>\n{content}")
        
        prompt = "\n".join(parts) + "\n<|assistant|>\n"
        
        # Encode
        start_ids = tokenizer.encode(prompt).ids
        
        # Truncate if too long
        if len(start_ids) > block_size - max_new_tokens:
            start_ids = start_ids[-(block_size - max_new_tokens):]
        
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        
        # Decode — FIX: slice by token count, not string length (BPE can misalign)
        new_ids = y[0].tolist()[len(start_ids):]
        reply = tokenizer.decode(new_ids)
        
        # Clean up stop tokens
        for stop in ["<|endoftext|>", "<|user|>", "<|padding|>"]:
            if stop in reply:
                reply = reply[:reply.index(stop)]
        
        reply = reply.strip()
        
        # Store in history
        conversation_history.append(("assistant", reply))
        
        print(f"\nAI: {reply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with GPT model")
    parser.add_argument("--checkpoint", type=str, default="out/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    
    chat(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
