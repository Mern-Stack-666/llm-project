# LLM From Scratch

A complete pipeline to build, train, fine-tune, and serve a custom GPT language model — supporting **20+ datasets** covering web text, **358 programming languages**, **100+ human languages**, books, and academic papers.

## Roadmap

- [x] **STEP 1**: Tokenizer — BPE tokenizer (32k vocab, code + multilingual)
- [x] **STEP 2**: Dataset Collection — Web scraper + 20+ HuggingFace datasets
- [x] **STEP 3**: Dataset Cleaning — Source marker removal, dedup, noise filtering
- [x] **STEP 4**: Model Architecture — GPT-2 Transformer with Flash Attention
- [x] **STEP 5**: Pretraining — YAML config, gradient accumulation, mixed precision
- [x] **STEP 6**: Fine-tuning — Instruction tuning (Alpaca/ShareGPT)
- [x] **STEP 7**: Serving — FastAPI REST API with `/generate` and `/chat`

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_pipeline.bat
```

## Training on World-Wide Data

### All datasets at once (web + code + multilingual + books)
```powershell
.venv\Scripts\python.exe -m src.data.prepare --dataset all --streaming
.venv\Scripts\python.exe -m src.training.train --set init_from=scratch
```

### Pick specific datasets
```powershell
.venv\Scripts\python.exe -m src.data.prepare --dataset multi --datasets openwebtext,the_stack,wikipedia --streaming
```

### Available Datasets

| Category | Dataset | Size | Command |
|----------|---------|------|---------|
| **Web** | OpenWebText | ~38 GB | `--dataset openwebtext` |
| **Web** | C4 | ~350 GB | `--dataset c4 --streaming` |
| **Web** | FineWeb | ~10B tok | `--dataset fineweb` |
| **Web** | FineWeb-Edu | ~10B tok | `--dataset fineweb-edu` |
| **Web** | RedPajama | ~1B tok | `--dataset redpajama` |
| **Web** | SlimPajama | ~627B tok | `--dataset slimpajama --streaming` |
| **Web** | The Pile | ~800 GB | `--dataset the_pile --streaming` |
| **Code** | The Stack | ~3 TB / 358 langs | `--dataset the_stack --streaming` |
| **Code** | StarCoder | ~250 GB | `--dataset starcoderdata --streaming` |
| **Code** | CodeParrot | ~50 GB | `--dataset codeparrot --streaming` |
| **Code** | CodeSearchNet | ~2M funcs | `--dataset code_search_net` |
| **Language** | Wikipedia | ~20 GB | `--dataset wikipedia` |
| **Language** | mC4 | Multilingual | `--dataset mc4 --lang es` |
| **Language** | OSCAR | Multilingual | `--dataset oscar --lang fr` |
| **Language** | CC-100 | 100 languages | `--dataset cc100 --lang zh` |
| **Books** | BookCorpus | ~5 GB | `--dataset bookcorpus` |
| **Academic** | ArXiv | Papers | `--dataset arxiv` |
| **Q&A** | OpenOrca | ~4M pairs | `--dataset openorca` |
| **Test** | WikiText-103 | ~500 MB | `--dataset wikitext-103` |
| **Local** | Your data | — | `--dataset local` |

### Multilingual Training
```powershell
# Train on Spanish web data
.venv\Scripts\python.exe -m src.data.prepare --dataset mc4 --lang es --streaming

# Train on Chinese web data  
.venv\Scripts\python.exe -m src.data.prepare --dataset cc100 --lang zh --streaming

# Supported: en, es, fr, de, it, pt, ru, zh, ja, ko, ar, hi, + 35 more
```

### Limit tokens (for dev/testing)
```powershell
.venv\Scripts\python.exe -m src.data.prepare --dataset openwebtext --max-train-tokens 10000000
```

## Pipeline Commands

```powershell
# List all datasets
.venv\Scripts\python.exe -m src.data.prepare --list

# Scrape data
.venv\Scripts\python.exe -m src.data.scraper

# Train tokenizer (32k vocab)
.venv\Scripts\python.exe -m src.data.train_tokenizer --vocab-size 32000

# Train tokenizer from HuggingFace data (no local files needed)
.venv\Scripts\python.exe -m src.data.train_tokenizer --from-dataset openwebtext --sample-size 100000

# Prepare data
.venv\Scripts\python.exe -m src.data.prepare --dataset openwebtext

# Train
.venv\Scripts\python.exe -m src.training.train

# Fine-tune
.venv\Scripts\python.exe -m src.training.finetune

# Serve API
.venv\Scripts\python.exe -m src.serving.api

# Chat
.venv\Scripts\python.exe -m src.inference.chat

# Generate
.venv\Scripts\python.exe -m src.inference.generate "The future of AI is"
```

## Configuration

All hyperparameters in `config/default.yaml`. Override via CLI:
```powershell
.venv\Scripts\python.exe -m src.training.train --set max_iters=50000 learning_rate=3e-4
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check & model info |
| POST | `/generate` | Generate text from prompt |
| POST | `/chat` | Multi-turn chat |
| GET | `/docs` | Swagger UI |

## Project Structure

```
llm-project/
├── config/default.yaml           # All hyperparameters
├── data/
│   ├── raw/                      # Scraped text files
│   ├── processed/                # Tokenized binary files
│   └── tokenizer.json            # Trained BPE tokenizer
├── out/
│   ├── ckpt.pt                   # Pretrained checkpoint
│   └── finetuned/ckpt.pt         # Fine-tuned checkpoint
├── src/
│   ├── config.py                 # YAML config loader
│   ├── data/
│   │   ├── scraper.py            # Web scraper
│   │   ├── train_tokenizer.py    # BPE tokenizer (32k, multilingual)
│   │   └── prepare.py            # 20+ dataset support
│   ├── model/gpt.py              # GPT architecture
│   ├── training/
│   │   ├── train.py              # Pretraining
│   │   └── finetune.py           # Instruction tuning
│   ├── inference/
│   │   ├── generate.py           # Text generation
│   │   └── chat.py               # Interactive chat
│   └── serving/api.py            # FastAPI REST server
├── requirements.txt
└── run_pipeline.bat
```
