import os
import numpy as np
from tokenizers import Tokenizer

def prepare_data():
    # Load tokenizer
    tokenizer_path = os.path.join("data", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("Tokenizer not found. Please run train_tokenizer.py first.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Read raw data
    input_file = os.path.join("data", "raw", "scraped_data.txt")
    if not os.path.exists(input_file):
        print("Raw data not found. Please run scraper.py first.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")

    # Encode with tokenizer
    print("Tokenizing data...")
    encoded = tokenizer.encode(data)
    ids = encoded.ids
    print(f"Total tokens: {len(ids):,}")

    # Split into train/val
    n = len(ids)
    train_ids = ids[:int(n*0.9)]
    val_ids = ids[int(n*0.9):]
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))
    print("Saved train.bin and val.bin")

if __name__ == "__main__":
    prepare_data()
