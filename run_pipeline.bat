@echo off
echo ==========================================
echo      LLM Training Pipeline
echo ==========================================

echo [0/4] Installing Dependencies...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo [1/4] Running Scraper...
.venv\Scripts\python.exe -m src.data.scraper

echo [2/4] Training Tokenizer...
.venv\Scripts\python.exe -m src.data.train_tokenizer

echo [3/4] Preparing Data (Tokenizing & Binning)...
.venv\Scripts\python.exe -m src.data.prepare

echo [4/4] Starting Training...
.venv\Scripts\python.exe -m src.training.train

pause
