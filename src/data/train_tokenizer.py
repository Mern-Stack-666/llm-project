from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os

def train_tokenizer(input_file, vocab_size=10000, save_path="data/tokenizer.json"):
    """
    Trains a BPE tokenizer on the provided text file.
    """
    print(f"Training tokenizer on {input_file} with vocab size {vocab_size}...")
    
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Initialize a trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<|padding|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train the tokenizer
    tokenizer.train([input_file], trainer)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    # Ensure you have data/raw/scraped_data.txt or similar before running
    input_data = os.path.join("data", "raw", "scraped_data.txt")
    if os.path.exists(input_data):
        train_tokenizer(input_data)
    else:
        print(f"File {input_data} not found. Please run the scraper first.")
