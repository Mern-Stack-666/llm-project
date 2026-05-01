@echo off
echo ==========================================
echo      LLM Training Pipeline
echo ==========================================

echo [0/6] Installing Dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [1/6] Running Scraper (data collection)...
.venv\Scripts\python.exe -m src.data.scraper

echo [2/6] Training Tokenizer (32k vocab, code + multilingual)...
.venv\Scripts\python.exe -m src.data.train_tokenizer --vocab-size 32000

echo [3/6] Preparing Data (Tokenizing ^& Binning)...
REM Change the --dataset flag to train on different data:
REM   local        = your scraped data only
REM   openwebtext  = 38 GB web text
REM   all          = everything (web + code + multilingual + books)
REM   multi        = pick specific ones: --datasets openwebtext,the_stack,wikipedia
.venv\Scripts\python.exe -m src.data.prepare --dataset local

echo [4/6] Starting Pretraining...
.venv\Scripts\python.exe -m src.training.train --set init_from=scratch

echo [5/6] Fine-tuning (Instruction Tuning)...
.venv\Scripts\python.exe -m src.training.finetune

echo ==========================================
echo      Pipeline Complete!
echo ==========================================
echo.
echo To start the API server:
echo   .venv\Scripts\python.exe -m src.serving.api
echo.
echo To train on ALL internet data (web + code + languages):
echo   .venv\Scripts\python.exe -m src.data.prepare --dataset all --streaming
echo   .venv\Scripts\python.exe -m src.training.train --set init_from=scratch
echo.
pause
