"""
Unified Data Preparation — Supports 20+ datasets covering:
  - General web text (OpenWebText, C4, FineWeb, RedPajama, The Pile)
  - Code in ALL languages (The Stack, StarCoder, CodeParrot)
  - Human languages (mc4, OSCAR, CC-100, Wikipedia multilingual)
  - Books, academic papers, Q&A, math, science

Usage:
    python -m src.data.prepare --dataset openwebtext
    python -m src.data.prepare --dataset the_stack --streaming
    python -m src.data.prepare --dataset mc4 --lang es --streaming
    python -m src.data.prepare --dataset multi --datasets openwebtext,the_stack,wikipedia
    python -m src.data.prepare --dataset all --streaming  # EVERYTHING
"""

import os
import re
import argparse
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Registry — every dataset we support
# ──────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    # ── General Web Text ──
    "openwebtext": {
        "path": "openwebtext", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~38 GB Reddit-curated web text",
    },
    "wikitext-103": {
        "path": "wikitext", "name": "wikitext-103-raw-v1", "text_key": "text",
        "split_train": "train", "split_val": "validation",
        "desc": "~500 MB Wikipedia (good for testing)",
    },
    "c4": {
        "path": "allenai/c4", "name": "en", "text_key": "text",
        "split_train": "train", "split_val": "validation",
        "desc": "~350 GB Colossal Clean Crawled Corpus",
    },
    "fineweb": {
        "path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~10B tokens curated web (HuggingFace FineWeb sample)",
    },
    "fineweb-edu": {
        "path": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~10B tokens educational web content",
    },
    "redpajama": {
        "path": "togethercomputer/RedPajama-Data-1T-Sample", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~1B token sample of RedPajama (LLaMA reproduction)",
    },
    "slimpajama": {
        "path": "cerebras/SlimPajama-627B", "text_key": "text",
        "split_train": "train", "split_val": "validation",
        "desc": "~627B tokens cleaned RedPajama",
    },
    "the_pile": {
        "path": "monology/pile-uncopyrighted", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~800 GB diverse text (books, arxiv, github, stackexchange)",
    },

    # ── Code / Programming Languages ──
    "the_stack": {
        "path": "bigcode/the-stack-dedup", "text_key": "content",
        "split_train": "train", "split_val": None,
        "desc": "~3 TB code in 358 programming languages",
    },
    "starcoderdata": {
        "path": "bigcode/starcoderdata", "text_key": "content",
        "split_train": "train", "split_val": None,
        "desc": "~250 GB curated code (StarCoder training data)",
    },
    "code_search_net": {
        "path": "code_search_net", "name": "all", "text_key": "whole_func_string",
        "split_train": "train", "split_val": "validation",
        "desc": "~2M code functions with docstrings (6 languages)",
    },
    "codeparrot": {
        "path": "codeparrot/codeparrot-clean", "text_key": "content",
        "split_train": "train", "split_val": None,
        "desc": "~50 GB cleaned GitHub Python code",
    },

    # ── Human Languages (Multilingual) ──
    "wikipedia": {
        "path": "wikipedia", "name": "20220301.en", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "Full English Wikipedia (~20 GB)",
    },
    "mc4": {
        "path": "mc4", "name": "en", "text_key": "text",
        "split_train": "train", "split_val": "validation",
        "desc": "Multilingual C4 (use --lang for other languages)",
    },
    "oscar": {
        "path": "oscar-corpus/OSCAR-2301", "name": "en", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "OSCAR multilingual web corpus (use --lang for others)",
    },
    "cc100": {
        "path": "cc100", "name": "en", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "Common Crawl 100 languages (use --lang)",
    },

    # ── Books, Academic, Q&A ──
    "bookcorpus": {
        "path": "bookcorpus/bookcorpus", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "~5 GB unpublished books",
    },
    "arxiv": {
        "path": "togethercomputer/RedPajama-Data-1T-Sample", "text_key": "text",
        "split_train": "train", "split_val": None,
        "desc": "Academic papers (via RedPajama sample)",
    },
    "openorca": {
        "path": "Open-Orca/OpenOrca", "text_key": "response",
        "split_train": "train", "split_val": None,
        "desc": "~4M instruction-response pairs",
    },
}

# Languages supported by multilingual datasets (mc4, oscar, cc100)
SUPPORTED_LANGS = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar",
    "hi", "tr", "pl", "sv", "da", "fi", "no", "cs", "ro", "hu", "el", "he",
    "th", "vi", "id", "ms", "uk", "bg", "hr", "sk", "sl", "et", "lv", "lt",
    "bn", "ta", "te", "ml", "mr", "gu", "kn", "pa", "ur", "fa", "sw",
]


def list_datasets():
    """Print all available datasets."""
    print("\n" + "=" * 70)
    print("  AVAILABLE DATASETS")
    print("=" * 70)
    categories = {
        "General Web": ["openwebtext", "c4", "fineweb", "fineweb-edu", "redpajama", "slimpajama", "the_pile", "wikitext-103"],
        "Code (All Languages)": ["the_stack", "starcoderdata", "codeparrot", "code_search_net"],
        "Human Languages": ["wikipedia", "mc4", "oscar", "cc100"],
        "Books & Academic": ["bookcorpus", "arxiv", "openorca"],
    }
    for cat, names in categories.items():
        print(f"\n  {cat}:")
        for n in names:
            d = DATASET_REGISTRY.get(n, {})
            print(f"    {n:20s} — {d.get('desc', '')}")
    print(f"\n  Special modes:")
    print(f"    {'multi':20s} — Combine multiple datasets (--datasets a,b,c)")
    print(f"    {'all':20s} — Train on EVERYTHING (streaming required)")
    print(f"    {'local':20s} — Your own data from data/raw/")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_hf_dataset(name, streaming=False, lang=None):
    """Load a HuggingFace dataset with appropriate config."""
    from datasets import load_dataset

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Use --list to see all.")

    cfg = dict(DATASET_REGISTRY[name])  # copy

    # Override language for multilingual datasets
    if lang and name in ("mc4", "oscar", "cc100"):
        cfg["name"] = lang
        print(f"  Using language: {lang}")

    load_args = {"path": cfg["path"], "streaming": streaming, "trust_remote_code": True}
    if "name" in cfg:
        load_args["name"] = cfg["name"]

    print(f"Loading '{name}': {cfg.get('desc', '')}")
    if streaming:
        print("  (streaming mode)")

    dataset = load_dataset(**load_args)
    return dataset, cfg


def load_local_dataset(raw_dir):
    """Load local text files."""
    texts = []
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
        if os.path.isfile(fpath) and fname.endswith(('.txt', '.md', '.py', '.js', '.ts', '.java', '.c', '.cpp', '.rs', '.go')):
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            text = re.sub(r'---\s*Source:.*?---', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            texts.append(text.strip())
    if not texts:
        raise FileNotFoundError(f"No text/code files found in {raw_dir}")
    return '\n\n'.join(texts)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization & Export
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_and_export(tokenizer, texts_iter, output_path, text_key=None,
                        chunk_size=10000, max_tokens=None, desc="Tokenizing",
                        append=False):
    """Tokenize iterable of strings/dicts and write to binary file."""
    total_tokens = 0
    mode = 'ab' if append else 'wb'

    with open(output_path, mode) as f:
        chunk = []
        for text in tqdm(texts_iter, desc=desc):
            if isinstance(text, dict):
                text = text.get(text_key or "text", "")
            if not text or not isinstance(text, str) or len(text.strip()) < 10:
                continue

            chunk.append(text)

            if len(chunk) >= chunk_size:
                combined = "\n".join(chunk)
                ids = tokenizer.encode(combined).ids
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                total_tokens += len(ids)
                chunk = []
                if max_tokens and total_tokens >= max_tokens:
                    print(f"  Reached max_tokens limit ({max_tokens:,})")
                    break

        if chunk:
            combined = "\n".join(chunk)
            ids = tokenizer.encode(combined).ids
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            total_tokens += len(ids)

    print(f"  {desc} done: {total_tokens:,} tokens → {output_path}")
    return total_tokens


def tokenize_text_block(tokenizer, text, output_path, desc="Tokenizing"):
    """Tokenize a single large text block."""
    print(f"  {desc}...")
    ids = tokenizer.encode(text).ids
    np.array(ids, dtype=np.uint16).tofile(output_path)
    print(f"  {desc} done: {len(ids):,} tokens → {output_path}")
    return len(ids)


# ──────────────────────────────────────────────────────────────────────────────
# Process a single dataset
# ──────────────────────────────────────────────────────────────────────────────

def process_single_dataset(name, tokenizer, train_path, val_path, streaming,
                           lang, max_train_tokens, max_val_tokens, val_ratio,
                           append=False):
    """Download, tokenize, and write one dataset to train/val bins."""

    if name == "local":
        text = load_local_dataset("data/raw")
        print(f"  Characters: {len(text):,}")
        split_idx = int(len(text) * (1.0 - val_ratio))
        tokenize_text_block(tokenizer, text[:split_idx], train_path, "Train")
        tokenize_text_block(tokenizer, text[split_idx:], val_path, "Val")
        return

    dataset, cfg = load_hf_dataset(name, streaming=streaming, lang=lang)
    text_key = cfg["text_key"]

    if cfg.get("split_val"):
        train_split = dataset[cfg["split_train"]]
        val_split = dataset[cfg["split_val"]]

        if streaming:
            train_iter = (item for item in train_split)
            val_iter = (item for item in val_split)
        else:
            train_iter = iter(train_split)
            val_iter = iter(val_split)

        tokenize_and_export(tokenizer, train_iter, train_path, text_key=text_key,
                            max_tokens=max_train_tokens, desc=f"Train({name})", append=append)
        tokenize_and_export(tokenizer, val_iter, val_path, text_key=text_key,
                            max_tokens=max_val_tokens or 500_000, desc=f"Val({name})", append=append)
    else:
        full_split = dataset[cfg["split_train"]]
        if streaming:
            val_count = 5000
            print(f"  Streaming: first {val_count} docs → val, rest → train")
            val_items, train_items = [], []
            for i, item in enumerate(tqdm(full_split, desc="Splitting")):
                if i < val_count:
                    val_items.append(item)
                else:
                    train_items.append(item)
                    if max_train_tokens and len(train_items) > max_train_tokens // 300:
                        break
            tokenize_and_export(tokenizer, val_items, val_path, text_key=text_key,
                                max_tokens=max_val_tokens, desc=f"Val({name})", append=append)
            tokenize_and_export(tokenizer, train_items, train_path, text_key=text_key,
                                max_tokens=max_train_tokens, desc=f"Train({name})", append=append)
        else:
            n = len(full_split)
            val_size = max(int(n * val_ratio), 1000)
            print(f"  Split: {n - val_size:,} train / {val_size:,} val")
            indices = np.random.default_rng(42).permutation(n)
            val_texts = (full_split[int(i)] for i in indices[:val_size])
            tokenize_and_export(tokenizer, val_texts, val_path, text_key=text_key,
                                max_tokens=max_val_tokens, desc=f"Val({name})", append=append)
            train_texts = (full_split[int(i)] for i in indices[val_size:])
            tokenize_and_export(tokenizer, train_texts, train_path, text_key=text_key,
                                max_tokens=max_train_tokens, desc=f"Train({name})", append=append)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

# Default datasets for "all" mode — the best mix for universal knowledge
ALL_DATASETS = [
    "fineweb",       # Clean web text
    "wikipedia",     # Encyclopedic knowledge
    "the_stack",     # Code in all languages
    "redpajama",     # Diverse (books, arxiv, stackexchange)
    "mc4",           # Multilingual web
    "bookcorpus",    # Books
]


def prepare_data(dataset_name, tokenizer_path, output_dir, streaming,
                 lang, max_train_tokens, max_val_tokens, val_ratio,
                 multi_datasets=None):
    """Master data preparation function."""

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        print("Run: python -m src.data.train_tokenizer")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizer: {tokenizer.get_vocab_size():,} vocab")

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    if dataset_name == "all":
        datasets_to_process = ALL_DATASETS
        print(f"\n  ALL MODE — training on {len(datasets_to_process)} datasets:")
        for d in datasets_to_process:
            print(f"    • {d}: {DATASET_REGISTRY[d]['desc']}")
        print()
    elif dataset_name == "multi":
        if not multi_datasets:
            print("ERROR: --datasets required for 'multi' mode (e.g. --datasets openwebtext,the_stack)")
            return
        datasets_to_process = [d.strip() for d in multi_datasets.split(",")]
        print(f"\n  MULTI MODE — {len(datasets_to_process)} datasets: {datasets_to_process}\n")
    else:
        datasets_to_process = [dataset_name]

    # Per-dataset token budget for multi/all mode
    if len(datasets_to_process) > 1 and max_train_tokens:
        per_ds_tokens = max_train_tokens // len(datasets_to_process)
        per_ds_val = (max_val_tokens or 500_000) // len(datasets_to_process)
    else:
        per_ds_tokens = max_train_tokens
        per_ds_val = max_val_tokens

    for i, ds_name in enumerate(datasets_to_process):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(datasets_to_process)}] Processing: {ds_name}")
        print(f"{'='*60}")
        try:
            process_single_dataset(
                ds_name, tokenizer, train_path, val_path,
                streaming=streaming, lang=lang,
                max_train_tokens=per_ds_tokens,
                max_val_tokens=per_ds_val,
                val_ratio=val_ratio,
                append=(i > 0),  # append after first dataset
            )
        except Exception as e:
            print(f"  ERROR processing {ds_name}: {e}")
            print(f"  Skipping and continuing...")
            continue

    # Report
    train_size = os.path.getsize(train_path) if os.path.exists(train_path) else 0
    val_size = os.path.getsize(val_path) if os.path.exists(val_path) else 0
    print(f"\n{'='*60}")
    print(f"  DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  train.bin : {train_size:,} bytes ({train_size // 2:,} tokens)")
    print(f"  val.bin   : {val_size:,} bytes ({val_size // 2:,} tokens)")
    ratio = val_size / max(train_size, 1) * 100
    print(f"  val/train : {ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--dataset", type=str, default="openwebtext",
                        help="Dataset name, 'multi', 'all', or 'local'. Use --list to see all.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names for 'multi' mode")
    parser.add_argument("--lang", type=str, default=None,
                        help="Language code for multilingual datasets (e.g. es, fr, zh)")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--streaming", action="store_true",
                        help="Streaming mode for huge datasets")
    parser.add_argument("--max-train-tokens", type=int, default=None,
                        help="Max training tokens (total across all datasets)")
    parser.add_argument("--max-val-tokens", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--list", action="store_true", help="List all available datasets")

    args = parser.parse_args()

    if args.list:
        list_datasets()
    else:
        prepare_data(
            dataset_name=args.dataset,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            streaming=args.streaming,
            lang=args.lang,
            max_train_tokens=args.max_train_tokens,
            max_val_tokens=args.max_val_tokens,
            val_ratio=args.val_ratio,
            multi_datasets=args.datasets,
        )
