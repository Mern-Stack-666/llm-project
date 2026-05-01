"""
FastAPI Serving API for GPT Model

Exposes REST endpoints for text generation and interactive chat.

Endpoints:
    GET  /              → Health check & model info
    POST /generate      → Generate text from a prompt
    POST /chat          → Multi-turn chat completion
    GET  /docs          → Auto-generated Swagger UI

Usage:
    python -m src.serving.api
    python -m src.serving.api --checkpoint out/finetuned/ckpt.pt --port 8000
    
    # Or via uvicorn directly:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import argparse
from typing import List, Optional

import torch
from tokenizers import Tokenizer

from src.model.gpt import GPT, GPTConfig

# ──────────────────────────────────────────────────────────────────────────────
# Lazy globals — initialized on startup
# ──────────────────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None


def _to_python(val):
    """Convert torch.Tensor scalars to native Python types so Pydantic can serialize them."""
    if hasattr(val, 'item'):   # covers torch.Tensor of any dtype
        return val.item()
    return val
_device = None
_model_info = {}


def load_model(checkpoint_path: str, tokenizer_path: str, device: str = "auto"):
    """Load model and tokenizer into global state."""
    global _model, _tokenizer, _device, _model_info
    
    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    else:
        _device = device
    
    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path} on {_device}...")
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    _model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    _model.load_state_dict(state_dict)
    _model.eval()
    _model.to(_device)
    
    _model_info = {
        "checkpoint": checkpoint_path,
        "device": _device,
        "parameters": f"{_model.get_num_params() / 1e6:.2f}M",
        "vocab_size": _to_python(model_args.get("vocab_size", "unknown")),
        "block_size": _to_python(model_args.get("block_size", "unknown")),
        "n_layer": _to_python(model_args.get("n_layer", "unknown")),
        "n_head": _to_python(model_args.get("n_head", "unknown")),
        "n_embd": _to_python(model_args.get("n_embd", "unknown")),
        "iter_num": _to_python(checkpoint.get("iter_num", "unknown")),
        "best_val_loss": _to_python(checkpoint.get("best_val_loss", "unknown")),
    }
    
    print(f"Model loaded: {_model_info['parameters']} parameters")
    print(f"  Architecture: {model_args.get('n_layer')}L / {model_args.get('n_head')}H / {model_args.get('n_embd')}E")


@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    stop_tokens: list = None,
) -> dict:
    """
    Generate text from a prompt.
    
    Returns dict with generated text, token count, and timing info.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    t0 = time.time()
    
    # Encode
    encoded = _tokenizer.encode(prompt)
    input_ids = torch.tensor(encoded.ids, dtype=torch.long, device=_device).unsqueeze(0)
    
    # Crop if too long
    block_size = _model.config.block_size
    if input_ids.size(1) > block_size:
        input_ids = input_ids[:, -block_size:]
    
    # Generate
    output_ids = _model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    
    # Decode — FIX: slice by token count, not string length (BPE spacing can cause misalignment)
    generated_ids = output_ids[0].tolist()
    prompt_len = len(encoded.ids)
    new_ids = generated_ids[prompt_len:]
    new_text = _tokenizer.decode(new_ids)
    full_text = _tokenizer.decode(generated_ids)
    
    # Handle stop tokens
    if stop_tokens:
        for st in stop_tokens:
            if st in new_text:
                new_text = new_text[:new_text.index(st)]
                break
    
    elapsed = time.time() - t0
    new_token_count = len(generated_ids) - len(encoded.ids)
    
    return {
        "prompt": prompt,
        "generated_text": new_text.strip(),
        "full_text": full_text,
        "tokens_generated": new_token_count,
        "time_seconds": round(elapsed, 3),
        "tokens_per_second": round(new_token_count / elapsed, 1) if elapsed > 0 else 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="GPT Language Model API",
    description="REST API for text generation with a custom-trained GPT model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    max_new_tokens: int = Field(200, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: float = Field(0.8, description="Sampling temperature", ge=0.1, le=2.0)
    top_k: int = Field(50, description="Top-k sampling", ge=1, le=500)


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    max_new_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(0.8, ge=0.1, le=2.0)
    top_k: int = Field(50, ge=1, le=500)


class ChatResponse(BaseModel):
    reply: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


class ModelInfoResponse(BaseModel):
    status: str
    model_info: dict


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_model=ModelInfoResponse)
def health_check():
    """Health check and model information."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        status="ok",
        model_info=_model_info,
    )


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """Generate text from a prompt."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
        return GenerateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Multi-turn chat completion."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format conversation history into a prompt
    parts = []
    for msg in req.messages:
        if msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}")
    
    # Add assistant prompt prefix so the model continues as assistant
    prompt = "\n".join(parts) + "\n<|assistant|>\n"
    
    try:
        result = generate(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            stop_tokens=["<|endoftext|>", "<|user|>"],
        )
        
        return ChatResponse(
            reply=result["generated_text"],
            tokens_generated=result["tokens_generated"],
            time_seconds=result["time_seconds"],
            tokens_per_second=result["tokens_per_second"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT Model Serving API")
    parser.add_argument("--checkpoint", type=str, default="out/ckpt.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu, mps")
    
    args = parser.parse_args()
    
    # Load model before starting server
    load_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )
    
    # Start server
    import uvicorn
    print(f"\n  Starting API server at http://{args.host}:{args.port}")
    print(f"  Swagger docs: http://localhost:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
