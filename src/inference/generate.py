import os
import torch
from tokenizers import Tokenizer
from src.model.gpt import GPT, GPTConfig

def generate_text(prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    # Load tokenizer
    tokenizer_path = os.path.join("data", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model
    ckpt_path = os.path.join("out", "ckpt.pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists(ckpt_path):
        print(f"Loading model from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    else:
        print("Checkpoint not found, initializing random model for testing...")
        # Default config matching train.py
        gptconf = GPTConfig(
            n_layer=12, n_head=12, n_embd=768, block_size=1024,
            vocab_size=tokenizer.get_vocab_size(), dropout=0.0
        )
        model = GPT(gptconf)

    model.eval()
    model.to(device)
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    print(f"\nGenerating from prompt: '{prompt}'\n" + "-"*50)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    import sys
    prompt = "Artificial intelligence is"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    
    generate_text(prompt)
