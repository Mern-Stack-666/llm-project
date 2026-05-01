"""
BPE Tokenizer Trainer — Multilingual & Code Aware

Trains a Byte-Pair Encoding tokenizer optimized for:
  - Natural language (English + multilingual)
  - Code in all programming languages
  - Special chat/instruction tokens

Usage:
    python -m src.data.train_tokenizer
    python -m src.data.train_tokenizer --vocab-size 32000
    python -m src.data.train_tokenizer --input data/raw/scraped_data.txt
    python -m src.data.train_tokenizer --from-dataset openwebtext --sample-size 100000
"""

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import os
import argparse
import glob
import tempfile


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|padding|>",
    "<|startoftext|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|code|>",
    "<|/code|>",
]


def train_tokenizer(input_files, vocab_size=32000, save_path="data/tokenizer.json"):
    """
    Train a BPE tokenizer on text files.
    
    Args:
        input_files: List of text file paths
        vocab_size: Vocabulary size (32k recommended for code+multilingual)
        save_path: Output path
    """
    print(f"Training BPE tokenizer: vocab_size={vocab_size}")
    print(f"  Files: {len(input_files)} file(s)")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tokenizer.train(input_files, trainer)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    tokenizer.save(save_path)

    print(f"\nTokenizer saved to {save_path}")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")

    # Test with English, code, and multilingual samples
    tests = [
        "Artificial intelligence is a branch of computer science.",
        "def hello_world():\n    print('Hello, World!')",
        "La inteligencia artificial es una rama de la informatica.",
    ]
    for t in tests:
        enc = tokenizer.encode(t)
        print(f"\n  '{t[:50]}...'")
        # FIX: guard against IndexError if sample tokenizes to fewer than 10 tokens
        preview = enc.tokens[:min(10, len(enc.tokens))]
        print(f"    → {len(enc.ids)} tokens: {preview}...")


def train_from_hf_dataset(dataset_name, sample_size, vocab_size, save_path, lang=None):
    """Train tokenizer directly from a HuggingFace dataset sample."""
    from datasets import load_dataset

    print(f"Downloading {sample_size:,} samples from '{dataset_name}' for tokenizer training...")

    load_args = {"path": dataset_name, "streaming": True, "trust_remote_code": True, "split": "train"}
    if lang:
        load_args["name"] = lang

    dataset = load_dataset(**load_args)

    # Collect text samples into a temp file
    tmp_path = os.path.join("data", "raw", "_tokenizer_train_data.txt")
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    text_key = "text"
    # Some datasets use "content" (code datasets)
    first_item = next(iter(dataset))
    if "content" in first_item and "text" not in first_item:
        text_key = "content"

    with open(tmp_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            if i >= sample_size:
                break
            text = item.get(text_key, "")
            if text and len(text.strip()) > 10:
                f.write(text + "\n")
            if i % 10000 == 0 and i > 0:
                print(f"  Collected {i:,}/{sample_size:,} samples...")

    print(f"  Saved {sample_size:,} samples to {tmp_path}")
    train_tokenizer([tmp_path], vocab_size=vocab_size, save_path=save_path)

    # Cleanup
    os.remove(tmp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer")
    parser.add_argument("--input", type=str, nargs="*", default=None,
                        help="Input text files (default: all in data/raw/)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="Vocabulary size (default: 32000)")
    parser.add_argument("--save", type=str, default="data/tokenizer.json")
    parser.add_argument("--from-dataset", type=str, default=None,
                        help="Train from a HuggingFace dataset (e.g. openwebtext)")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Number of samples from HF dataset (default: 100k)")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code for multilingual datasets")

    args = parser.parse_args()

    if args.from_dataset:
        train_from_hf_dataset(
            args.from_dataset, args.sample_size,
            args.vocab_size, args.save, args.lang
        )
    else:
        input_files = args.input or glob.glob(os.path.join("data", "raw", "*.txt"))
        if not input_files:
            print("No input files found. Options:")
            print("  1. Run scraper first:  python -m src.data.scraper")
            print("  2. Train from HF:     python -m src.data.train_tokenizer --from-dataset openwebtext")
        else:
            train_tokenizer(input_files, vocab_size=args.vocab_size, save_path=args.save)
