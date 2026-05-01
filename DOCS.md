# LLM Project — Complete Documentation

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Setup](#setup)
3. [Training Types](#training-types)
4. [Data Preparation — All Flags](#data-preparation)
5. [Tokenizer Training — All Flags](#tokenizer-training)
6. [Pretraining — All Flags](#pretraining)
7. [Fine-tuning — All Flags](#fine-tuning)
8. [Inference — All Flags](#inference)
9. [Serving API — All Flags](#serving-api)
10. [Configuration System](#configuration-system)
11. [All 20+ Datasets](#all-datasets)
12. [Model Architecture](#model-architecture)
13. [Training Recipes](#training-recipes)
14. [Hardware Requirements](#hardware-requirements)

---

## Pipeline Overview

```
Scraper → Tokenizer → Data Prep → Pretraining → Fine-tuning → Serving API
  (1)       (2)         (3)          (4)           (5)           (6)
```

Each step can be run independently. Run everything at once with `run_pipeline.bat`.

---

## What Each Pipeline Step Does

### Step 1: Scraper — Collecting Raw Text

**What it is:** A web crawler that downloads text from websites and saves it as plain `.txt` files.

**Why it matters:** An AI model learns by reading text — the more diverse, high-quality text you feed it, the smarter it gets. The scraper is how you gather that raw material.

**What happens under the hood:**
1. Sends HTTP requests to a list of URLs (Wikipedia articles by default)
2. Parses the HTML and extracts only `<p>` paragraph text (skips menus, footers, ads)
3. Cleans the text: removes citation brackets `[1]`, excessive whitespace, navigation junk
4. Appends everything into `data/raw/scraped_data.txt`

**Think of it like:** Going to a library and photocopying every book — you're collecting the raw reading material.

```powershell
python -m src.data.scraper                  # scrape default URLs
python -m src.data.scraper --clean-only     # just clean existing data
```

---

### Step 2: Tokenizer — Building a Vocabulary

**What it is:** A tokenizer converts human-readable text into numbers (tokens) that the model can process. It also builds a "vocabulary" — a dictionary of all the sub-words the model knows.

**Why it matters:** Computers don't understand words — they only understand numbers. The tokenizer is the translator between human language and the model's language. A good tokenizer compresses text efficiently (fewer tokens = faster training).

**What happens under the hood:**
1. Reads all your text files
2. Uses **Byte-Pair Encoding (BPE)** — starts with individual characters, then repeatedly merges the most common pairs: `t + h → th`, `th + e → the`
3. Builds a vocabulary of 32,000 sub-word tokens that can represent any text
4. Saves the vocabulary to `data/tokenizer.json`

**Example:**
```
Input:  "Artificial intelligence is amazing"
Output: [4521, 8903, 267, 15234]  ← 4 token IDs
```

**Think of it like:** Creating a codebook. Instead of writing "the" every time, you assign it number `267`. The model only works with these numbers.

```powershell
python -m src.data.train_tokenizer                                  # from local files
python -m src.data.train_tokenizer --from-dataset openwebtext       # from internet data
```

---

### Step 3: Data Preparation — Converting Text to Training Data

**What it is:** Takes raw text, runs it through the tokenizer, and saves the result as compact binary files (`.bin`) that the training loop can read at maximum speed.

**Why it matters:** You can't feed raw text files to GPU training — it would be extremely slow. Binary files loaded via memory-mapping (`np.memmap`) let the GPU fetch data instantly without loading everything into RAM.

**What happens under the hood:**
1. Downloads or loads the dataset (OpenWebText, The Stack, Wikipedia, etc.)
2. Tokenizes every document into sequences of token IDs
3. Saves as `train.bin` (90-95% of data) and `val.bin` (5-10% held out for evaluation)
4. Files are stored as `uint16` arrays — each token is just 2 bytes

**Output files:**
```
data/processed/train.bin  → Training data (model learns from this)
data/processed/val.bin    → Validation data (model is tested on this, never trains on it)
```

**Think of it like:** Converting a library of books into a massive spreadsheet of numbers that a computer can speed-read.

```powershell
python -m src.data.prepare --dataset openwebtext                    # English web text
python -m src.data.prepare --dataset all --streaming                # EVERYTHING
python -m src.data.prepare --dataset the_stack --streaming          # all code languages
python -m src.data.prepare --dataset mc4 --lang es --streaming      # Spanish
```

---

### Step 4: Pretraining — Teaching the Model to Understand Language

**What it is:** The core training process where the model reads billions of tokens and learns patterns — grammar, facts, reasoning, code syntax, and more. This is where the magic happens.

**Why it matters:** This is what makes a "language model." Without pretraining, the model is just random numbers. After pretraining, it can predict the next word in any sentence — which means it "understands" language.

**How it works (Next-Token Prediction):**
```
Input:  "The capital of France is"
Target: "Paris"

The model sees the input, predicts a word, compares it to the target,
and adjusts its weights to get closer next time. Repeat billions of times.
```

**What happens under the hood:**
1. Randomly initializes ~50M parameters (weights)
2. Grabs a batch of token sequences from `train.bin`
3. For each sequence, the model predicts the next token at every position
4. Computes the **loss** (how wrong the predictions were)
5. Uses **backpropagation** to calculate how to adjust each weight
6. Updates weights using the **AdamW optimizer**
7. Repeats for 20,000+ iterations
8. Every 100 steps, checks performance on `val.bin` and saves a checkpoint

**Key concepts:**
- **Loss:** A number measuring how bad the model is. Starts high (~10), should decrease to ~3-4. Lower = better.
- **Learning Rate:** How big each weight update is. Too high = unstable, too low = slow learning.
- **Checkpoint:** A saved snapshot of the model (`out/ckpt.pt`). You can resume training from any checkpoint.

**Think of it like:** A student reading millions of books and learning patterns — what word typically comes next, how sentences are structured, what facts are commonly stated.

```powershell
python -m src.training.train --set init_from=scratch               # start fresh
python -m src.training.train --set init_from=resume                 # continue training
python -m src.training.train --set init_from=scratch max_iters=50000 learning_rate=3e-4
```

---

### Step 5: Fine-tuning — Teaching the Model to Follow Instructions

**What it is:** After pretraining, the model can predict text but doesn't know how to answer questions or follow commands. Fine-tuning takes the pretrained model and teaches it conversation skills using instruction-response pairs.

**Why it matters:** This is the difference between a model that completes random text vs. one that actually answers your questions like ChatGPT. Pretraining gives knowledge; fine-tuning gives behavior.

**Before vs. After Fine-tuning:**
```
BEFORE (pretrained only):
  Input:  "What is Python?"
  Output: "What is Python used for in modern applications..."  ← just continues the text

AFTER (fine-tuned):
  Input:  "What is Python?"
  Output: "Python is a high-level programming language known for its simplicity..."  ← actually answers
```

**What happens under the hood:**
1. Loads the pretrained checkpoint (`out/ckpt.pt`)
2. Downloads an instruction dataset (e.g., Alpaca — 52k instruction-response pairs)
3. Formats each example with special tokens: `<|user|>` question `<|assistant|>` answer
4. Trains with a much **lower learning rate** (2e-5 vs. 1e-3) to gently adjust without destroying the knowledge learned during pretraining
5. Uses **higher dropout** (0.1) to prevent overfitting on the smaller dataset
6. Saves the fine-tuned model to `out/finetuned/ckpt.pt`

**Supported instruction formats:**
- **Alpaca:** `{"instruction": "Explain gravity", "output": "Gravity is a force..."}`
- **ShareGPT:** Multi-turn conversations `[{from: "human", value: "..."}, {from: "gpt", value: "..."}]`
- **Text:** Any plain text dataset

**Think of it like:** A medical student who has read all textbooks (pretraining) now doing residency at a hospital — learning how to talk to patients and give clear answers (fine-tuning).

```powershell
python -m src.training.finetune                                     # default (Alpaca)
python -m src.training.finetune --set finetune.dataset=Open-Orca/OpenOrca
python -m src.training.finetune --set finetune.max_iters=10000
```

---

### Step 6: Serving API — Making Your Model Available to the World

**What it is:** A REST API web server that loads your trained model and exposes it as HTTP endpoints — any app, website, or script can send a request and get generated text back.

**Why it matters:** Without an API, you can only use the model from the command line on your own machine. With the API, you can build apps on top of it, share it with others, or integrate it into any software.

**What happens under the hood:**
1. Loads the model checkpoint and tokenizer into memory
2. Starts a **FastAPI** web server (default: `http://localhost:8000`)
3. Listens for incoming HTTP requests
4. When a request arrives at `/generate` or `/chat`:
   - Tokenizes the input prompt
   - Runs the model to generate new tokens
   - Decodes tokens back to text
   - Returns the response as JSON
5. Auto-generates interactive **Swagger documentation** at `/docs`

**Endpoints:**
| Endpoint | What it does |
|----------|-------------|
| `GET /` | Health check — confirms model is loaded, shows model info |
| `POST /generate` | Send a prompt, get generated text back |
| `POST /chat` | Send conversation history, get assistant reply |
| `GET /docs` | Interactive API documentation (try it in your browser) |

**Think of it like:** Building a drive-through window for your restaurant (model). People don't need to come into the kitchen — they just place an order (request) and get food (response).

```powershell
python -m src.serving.api                                           # serve pretrained model
python -m src.serving.api --checkpoint out/finetuned/ckpt.pt        # serve fine-tuned model
python -m src.serving.api --port 3000 --device cuda                 # custom port + GPU
```

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Training Types

You can train your model in **4 different ways**:

### Type 1: Train from Scratch
Start a brand new model with random weights.
```powershell
.venv\Scripts\python.exe -m src.training.train --set init_from=scratch
```

### Type 2: Resume Training
Continue training from the last saved checkpoint.
```powershell
.venv\Scripts\python.exe -m src.training.train --set init_from=resume
```

### Type 3: Instruction Fine-tuning
Take a pretrained model and teach it to follow instructions (chat, Q&A).
```powershell
.venv\Scripts\python.exe -m src.training.finetune
```

### Type 4: Evaluate Only
Run evaluation on the validation set without any training.
```powershell
.venv\Scripts\python.exe -m src.training.train --set eval_only=true
```

---

## Data Preparation

**Module:** `python -m src.data.prepare`

### All Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | string | `openwebtext` | Dataset name. Options: any from the registry, `multi`, `all`, `local` |
| `--datasets` | string | None | Comma-separated list for `multi` mode (e.g. `openwebtext,the_stack`) |
| `--lang` | string | None | Language code for multilingual datasets (`es`, `fr`, `zh`, `ar`, etc.) |
| `--streaming` | flag | off | Stream data instead of downloading everything first. **Required** for huge datasets |
| `--max-train-tokens` | int | unlimited | Cap the number of training tokens |
| `--max-val-tokens` | int | unlimited | Cap the number of validation tokens |
| `--val-ratio` | float | `0.05` | Fraction of data used for validation (5%) |
| `--tokenizer` | string | `data/tokenizer.json` | Path to trained tokenizer |
| `--output-dir` | string | `data/processed` | Output directory for `.bin` files |
| `--list` | flag | off | Print all available datasets and exit |

### Dataset Modes

**Single dataset:**
```powershell
python -m src.data.prepare --dataset openwebtext
```

**Multiple datasets combined:**
```powershell
python -m src.data.prepare --dataset multi --datasets openwebtext,the_stack,wikipedia --streaming
```

**ALL datasets (web + code + multilingual + books):**
```powershell
python -m src.data.prepare --dataset all --streaming
```

**Local files only:**
```powershell
python -m src.data.prepare --dataset local
```

**Multilingual (Spanish, French, Chinese, etc.):**
```powershell
python -m src.data.prepare --dataset mc4 --lang es --streaming
python -m src.data.prepare --dataset cc100 --lang zh --streaming
```

**Quick dev run (limited tokens):**
```powershell
python -m src.data.prepare --dataset openwebtext --max-train-tokens 5000000
```

---

## Tokenizer Training

**Module:** `python -m src.data.train_tokenizer`

### All Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` | string(s) | all `.txt` in `data/raw/` | Input text files to train on |
| `--vocab-size` | int | `32000` | Vocabulary size. 32k is good for code + multilingual |
| `--save` | string | `data/tokenizer.json` | Output path |
| `--from-dataset` | string | None | Train tokenizer from a HuggingFace dataset (no local files needed) |
| `--sample-size` | int | `100000` | Number of samples when using `--from-dataset` |
| `--lang` | string | None | Language for multilingual HF datasets |

### Examples

```powershell
# From local files
python -m src.data.train_tokenizer --vocab-size 32000

# From HuggingFace (no local data needed)
python -m src.data.train_tokenizer --from-dataset openwebtext --sample-size 200000

# Smaller vocab for testing
python -m src.data.train_tokenizer --vocab-size 8000
```

### Special Tokens Included
`<|endoftext|>`, `<|padding|>`, `<|startoftext|>`, `<|user|>`, `<|assistant|>`, `<|system|>`, `<|code|>`, `<|/code|>`

---

## Pretraining

**Module:** `python -m src.training.train`

### All Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | string | `config/default.yaml` | Path to YAML config file |
| `--set` | key=val pairs | — | Override any config value inline |

### All Config Values (settable via `--set`)

#### I/O
| Key | Default | Description |
|-----|---------|-------------|
| `out_dir` | `out` | Directory for checkpoints |
| `eval_interval` | `100` | Evaluate every N steps |
| `log_interval` | `1` | Print loss every N steps |
| `eval_iters` | `50` | Batches averaged per evaluation |
| `eval_only` | `false` | Only evaluate, don't train |
| `always_save_checkpoint` | `true` | Save every eval, not just best |
| `init_from` | `resume` | `scratch` or `resume` |

#### Data
| Key | Default | Description |
|-----|---------|-------------|
| `data_dir` | `data/processed` | Where `train.bin`/`val.bin` live |
| `tokenizer_path` | `data/tokenizer.json` | Tokenizer file |

#### Batch / Sequence
| Key | Default | Description |
|-----|---------|-------------|
| `batch_size` | `12` | Sequences per micro-batch |
| `block_size` | `256` | Context window (max sequence length) |
| `gradient_accumulation_steps` | `4` | Micro-batches before optimizer step |

> **Effective batch size** = `batch_size × gradient_accumulation_steps` = 48 sequences

#### Model Architecture
| Key | Default | Description |
|-----|---------|-------------|
| `n_layer` | `12` | Transformer blocks |
| `n_head` | `6` | Attention heads |
| `n_embd` | `384` | Embedding dimension |
| `dropout` | `0.0` | Dropout rate (0 for pretraining) |
| `bias` | `false` | Use bias in Linear/LayerNorm |

#### Optimizer (AdamW)
| Key | Default | Description |
|-----|---------|-------------|
| `learning_rate` | `1e-3` | Peak learning rate |
| `max_iters` | `20000` | Total training steps |
| `weight_decay` | `0.1` | L2 regularization |
| `beta1` | `0.9` | Adam beta1 |
| `beta2` | `0.95` | Adam beta2 |
| `grad_clip` | `1.0` | Max gradient norm (0 = disabled) |

#### LR Schedule (Cosine with Warmup)
| Key | Default | Description |
|-----|---------|-------------|
| `decay_lr` | `true` | Enable LR decay |
| `warmup_iters` | `500` | Linear warmup steps |
| `lr_decay_iters` | `20000` | Steps until min_lr reached |
| `min_lr` | `1e-4` | Minimum learning rate |

#### System
| Key | Default | Description |
|-----|---------|-------------|
| `device` | `auto` | `auto`, `cuda`, `cpu`, `mps` |
| `dtype` | `auto` | `auto`, `float32`, `bfloat16`, `float16` |
| `compile` | `false` | PyTorch 2.0 `torch.compile` |

### Examples

```powershell
# Train from scratch, 50k steps, lower LR
python -m src.training.train --set init_from=scratch max_iters=50000 learning_rate=3e-4

# Bigger model
python -m src.training.train --set init_from=scratch n_layer=24 n_head=12 n_embd=768 block_size=512

# Use custom config
python -m src.training.train --config config/big_model.yaml

# CPU-only training
python -m src.training.train --set device=cpu dtype=float32
```

---

## Fine-tuning

**Module:** `python -m src.training.finetune`

### All Config Values (under `finetune.*` in YAML, or `--set finetune.key=val`)

| Key | Default | Description |
|-----|---------|-------------|
| `finetune.base_checkpoint` | `out/ckpt.pt` | Pretrained checkpoint to start from |
| `finetune.output_dir` | `out/finetuned` | Where to save fine-tuned checkpoint |
| `finetune.dataset` | `tatsu-lab/alpaca` | HuggingFace dataset for instructions |
| `finetune.dataset_split` | `train` | Which split to use |
| `finetune.max_iters` | `5000` | Fine-tuning steps |
| `finetune.learning_rate` | `2e-5` | LR (much lower than pretraining) |
| `finetune.dropout` | `0.1` | Dropout (higher than pretraining) |
| `finetune.warmup_iters` | `100` | Warmup steps |
| `finetune.eval_interval` | `50` | Eval frequency |
| `finetune.eval_iters` | `20` | Batches per eval |
| `finetune.batch_size` | `8` | Batch size |
| `finetune.gradient_accumulation_steps` | `4` | Grad accumulation |
| `finetune.grad_clip` | `1.0` | Gradient clipping |

### Supported Instruction Formats (auto-detected)

| Format | Fields | Example Dataset |
|--------|--------|-----------------|
| **Alpaca** | `instruction`, `input`, `output` | `tatsu-lab/alpaca` |
| **ShareGPT** | `conversations[{from, value}]` | `anon8231489123/ShareGPT_Vicuna_unfiltered` |
| **Text** | `text` | Any text dataset |

### Examples

```powershell
# Default (Alpaca dataset)
python -m src.training.finetune

# Different dataset
python -m src.training.finetune --set finetune.dataset=Open-Orca/OpenOrca

# More training steps
python -m src.training.finetune --set finetune.max_iters=10000 finetune.learning_rate=1e-5

# Fine-tune a specific checkpoint
python -m src.training.finetune --set finetune.base_checkpoint=out/ckpt.pt
```

---

## Inference

### Generate Text
**Module:** `python -m src.inference.generate`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | positional | `"Artificial intelligence is"` | Text prompt |
| `--checkpoint` | string | `out/ckpt.pt` | Model checkpoint |
| `--tokenizer` | string | `data/tokenizer.json` | Tokenizer path |
| `--max-tokens` | int | `200` | Max tokens to generate |
| `--temperature` | float | `0.8` | Randomness (lower = more deterministic) |
| `--top-k` | int | `50` | Top-k sampling |
| `--device` | string | `auto` | Device |

### Interactive Chat
**Module:** `python -m src.inference.chat`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | string | `out/ckpt.pt` | Model checkpoint |
| `--tokenizer` | string | `data/tokenizer.json` | Tokenizer path |
| `--max-tokens` | int | `150` | Max tokens per reply |
| `--temperature` | float | `0.8` | Randomness |
| `--top-k` | int | `40` | Top-k sampling |
| `--device` | string | `auto` | Device |

Chat commands: `exit`/`quit` to stop, `clear` to reset history.

```powershell
# Chat with pretrained model
python -m src.inference.chat

# Chat with fine-tuned model
python -m src.inference.chat --checkpoint out/finetuned/ckpt.pt
```

---

## Serving API

**Module:** `python -m src.serving.api`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | string | `out/ckpt.pt` | Model checkpoint |
| `--tokenizer` | string | `data/tokenizer.json` | Tokenizer path |
| `--host` | string | `0.0.0.0` | Server host |
| `--port` | int | `8000` | Server port |
| `--device` | string | `auto` | Device |

### Endpoints

#### `GET /` — Health Check
```json
{"status": "ok", "model_info": {"parameters": "29.50M", "device": "cuda", ...}}
```

#### `POST /generate` — Text Generation
```json
{
  "prompt": "Explain machine learning",
  "max_new_tokens": 200,
  "temperature": 0.8,
  "top_k": 50
}
```

#### `POST /chat` — Multi-turn Chat
```json
{
  "messages": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What is it used for?"}
  ],
  "max_new_tokens": 200,
  "temperature": 0.8,
  "top_k": 50
}
```

#### `GET /docs` — Swagger UI (auto-generated)

---

## Configuration System

All config lives in `config/default.yaml`. Three ways to customize:

### 1. Edit the YAML file
```yaml
# config/default.yaml
max_iters: 50000
learning_rate: 3.0e-4
n_layer: 24
```

### 2. Override via CLI
```powershell
python -m src.training.train --set max_iters=50000 learning_rate=3e-4
```

### 3. Use a custom config file
```powershell
python -m src.training.train --config config/big_model.yaml
```

Dot notation works for nested keys:
```powershell
python -m src.training.train --set finetune.learning_rate=1e-5 finetune.max_iters=8000
```

---

## All Datasets

### General Web Text
| Name | Size | Streaming? | Description |
|------|------|-----------|-------------|
| `openwebtext` | ~38 GB | Optional | Reddit-upvoted web links |
| `c4` | ~350 GB | **Required** | Colossal Clean Crawled Corpus |
| `fineweb` | ~10B tokens | Optional | HuggingFace curated web |
| `fineweb-edu` | ~10B tokens | Optional | Educational web content |
| `redpajama` | ~1B tokens | Optional | LLaMA reproduction sample |
| `slimpajama` | ~627B tokens | **Required** | Cleaned RedPajama |
| `the_pile` | ~800 GB | **Required** | Books + ArXiv + GitHub + StackExchange |
| `wikitext-103` | ~500 MB | No | Wikipedia subset (good for testing) |

### Code (358 Programming Languages)
| Name | Size | Languages | Description |
|------|------|-----------|-------------|
| `the_stack` | ~3 TB | 358 | Deduplicated GitHub code |
| `starcoderdata` | ~250 GB | 86 | StarCoder training data |
| `codeparrot` | ~50 GB | Python | Cleaned GitHub Python |
| `code_search_net` | ~2M funcs | 6 | Functions with docstrings |

### Human Languages (100+)
| Name | Langs | Flag Example | Description |
|------|-------|-------------|-------------|
| `wikipedia` | English | — | Full English Wikipedia |
| `mc4` | 100+ | `--lang es` | Multilingual C4 |
| `oscar` | 150+ | `--lang fr` | OSCAR web corpus |
| `cc100` | 100 | `--lang zh` | Common Crawl monolingual |

**Supported language codes:** `en`, `es`, `fr`, `de`, `it`, `pt`, `nl`, `ru`, `zh`, `ja`, `ko`, `ar`, `hi`, `tr`, `pl`, `sv`, `da`, `fi`, `no`, `cs`, `ro`, `hu`, `el`, `he`, `th`, `vi`, `id`, `ms`, `uk`, `bg`, `hr`, `sk`, `sl`, `et`, `lv`, `lt`, `bn`, `ta`, `te`, `ml`, `mr`, `gu`, `kn`, `pa`, `ur`, `fa`, `sw`

### Books & Academic
| Name | Size | Description |
|------|------|-------------|
| `bookcorpus` | ~5 GB | Unpublished books |
| `arxiv` | Varies | Academic papers |
| `openorca` | ~4M pairs | Instruction-response pairs |

---

## Model Architecture

GPT-2 style decoder-only Transformer.

| Component | Details |
|-----------|---------|
| Attention | Causal self-attention with Flash Attention (PyTorch ≥ 2.0) |
| FFN | GELU activation, 4× expansion |
| Normalization | Pre-LayerNorm (before attention and FFN) |
| Embeddings | Token + Positional, weight-tied with output head |
| Optimizer | AdamW with fused CUDA kernels when available |

### Model Size Presets

| Name | Layers | Heads | Embed | Params | `--set` flags |
|------|--------|-------|-------|--------|---------------|
| **Tiny** | 6 | 6 | 384 | ~29M | `n_layer=6 n_head=6 n_embd=384` |
| **Small** | 12 | 6 | 384 | ~50M | `n_layer=12 n_head=6 n_embd=384` |
| **Medium** | 12 | 12 | 768 | ~124M | `n_layer=12 n_head=12 n_embd=768` |
| **Large** | 24 | 16 | 1024 | ~350M | `n_layer=24 n_head=16 n_embd=1024` |
| **XL** | 36 | 20 | 1280 | ~774M | `n_layer=36 n_head=20 n_embd=1280` |

---

## Training Recipes

### Recipe 1: Quick Test (5 minutes, CPU)
```powershell
python -m src.data.prepare --dataset wikitext-103
python -m src.training.train --set init_from=scratch max_iters=500 device=cpu dtype=float32 batch_size=4
```

### Recipe 2: Local Data Only (your scraped data)
```powershell
python -m src.data.scraper
python -m src.data.train_tokenizer
python -m src.data.prepare --dataset local
python -m src.training.train --set init_from=scratch
```

### Recipe 3: Web Knowledge (single GPU, ~1 day)
```powershell
python -m src.data.prepare --dataset openwebtext
python -m src.training.train --set init_from=scratch max_iters=50000
```

### Recipe 4: Code Expert (single GPU)
```powershell
python -m src.data.prepare --dataset starcoderdata --streaming --max-train-tokens 50000000
python -m src.training.train --set init_from=scratch max_iters=30000
```

### Recipe 5: Multilingual (Spanish example)
```powershell
python -m src.data.prepare --dataset mc4 --lang es --streaming --max-train-tokens 50000000
python -m src.training.train --set init_from=scratch
```

### Recipe 6: Everything (web + code + languages + books)
```powershell
python -m src.data.prepare --dataset all --streaming
python -m src.training.train --set init_from=scratch max_iters=100000
python -m src.training.finetune
python -m src.serving.api --checkpoint out/finetuned/ckpt.pt
```

### Recipe 7: Bigger Model (requires good GPU)
```powershell
python -m src.data.prepare --dataset openwebtext
python -m src.training.train --set init_from=scratch n_layer=24 n_head=16 n_embd=1024 block_size=512 batch_size=4 gradient_accumulation_steps=16 max_iters=100000
```

---

## Hardware Requirements

| Recipe | GPU VRAM | RAM | Disk | Time |
|--------|----------|-----|------|------|
| Quick Test (CPU) | None | 4 GB | 1 GB | 5 min |
| Local Data | 4 GB+ | 8 GB | 2 GB | 30 min |
| OpenWebText | 8 GB+ | 16 GB | 50 GB | 1-2 days |
| Code Training | 8 GB+ | 16 GB | 50 GB+ | 1-2 days |
| Everything (all) | 16 GB+ | 32 GB | 200 GB+ | 1+ week |
| Bigger Model (350M+) | 24 GB+ | 32 GB | 200 GB+ | 1+ week |

> **Tip:** Use `--max-train-tokens` to limit data size if you're short on disk or time.
> Use `--streaming` to avoid downloading entire datasets upfront.
