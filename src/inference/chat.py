import os
import torch
from tokenizers import Tokenizer
from src.model.gpt import GPT, GPTConfig

def chat():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = 'out'
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Please train the model first.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Fix potential key issues from compilation/DDP
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(os.path.join('data', 'tokenizer.json'))
    
    print("\n" + "="*50)
    print("  TINY-GPT CHAT (Type 'exit' to stop)")
    print("="*50)

    while True:
        prompt = input("\nYou: ").strip()
        if not prompt:
            continue
        if prompt.lower() in ['exit', 'quit']:
            break
        
        # Encode the prompt
        start_ids = tokenizer.encode(prompt).ids
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

        # Generate!
        with torch.no_grad():
            # Generate 50 tokens
            y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
            
            # Decode and print
            full_text = tokenizer.decode(y[0].tolist())
            
            # Print only the generated part (optional, but here we show full)
            print("-" * 30)
            print(f"AI: {full_text}")
            print("-" * 30)

if __name__ == "__main__":
    chat()
