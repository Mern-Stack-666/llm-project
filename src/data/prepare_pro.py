import os
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer
from datasets import load_dataset # pip install datasets

def prepare_professional_data():
    """
    Downloads a professional dataset (e.g., WikiText-103) and tokenizes it.
    WikiText-103 is a standard benchmark dataset for LLMs.
    """
    
    # 1. Load Tokenizer
    tokenizer_path = os.path.join("data", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Please run train_tokenizer.py first.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 2. Download Dataset from Hugging Face
    print("Downloading WikiText-103 dataset (Professional Grade)...")
    # We use 'raw' to get the full text markers
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # WikiText has 'train', 'test', and 'validation' splits
    # We will combine them and then do our own 90/10 split or just use their split
    
    def process_split(split_name, output_path):
        print(f"Tokenizing {split_name} split...")
        data_list = dataset[split_name]['text']
        
        # Open file for binary writing
        with open(output_path, 'wb') as f:
            total_tokens = 0
            # Process in chunks of 50,000 lines to save memory
            chunk_size = 50000
            for i in tqdm(range(0, len(data_list), chunk_size)):
                chunk = data_list[i : i + chunk_size]
                # Filter out empty lines and join
                text = "\n".join([t for t in chunk if len(t.strip()) > 0])
                if not text:
                    continue
                
                # Tokenize this chunk
                ids = tokenizer.encode(text).ids
                # Write as uint16
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                total_tokens += len(ids)
        
        print(f"  Finished {split_name}. Total tokens: {total_tokens:,}")
        return total_tokens

    # 3. Export to binary files
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    # Process train and validation
    process_split('train', train_path)
    process_split('validation', val_path)
    
    print("\nSUCCESS!")
    print("Your model is now ready to be trained on 'World Knowledge' without crashing your RAM.")

if __name__ == "__main__":
    prepare_professional_data()
