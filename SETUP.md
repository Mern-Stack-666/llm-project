# The Ultimate, Step-by-Step Setup Guide for a New PC

This document provides **exhaustive, detailed instructions** to take a brand new Windows computer from absolutely zero to running your custom LLM project. Read every step carefully.

---

## Phase 1: Installing the Core Software (Prerequisites)

Before you touch any code, the new computer needs the right foundation.

### 1. Install Python (The Engine)
AI models run on Python. However, you cannot use just any version.
1. Go to [python.org/downloads/windows](https://www.python.org/downloads/windows/).
2. Scroll down and find **Python 3.10.11** or **Python 3.11.9**. *(Do not use Python 3.12 or 3.13, as many AI libraries like PyTorch are not fully stable on them yet).*
3. Download the **Windows installer (64-bit)**.
4. **CRITICAL STEP:** When you open the installer, before you click "Install Now", look at the very bottom of the window. You **MUST** check the box that says **"Add Python 3.x to PATH"**. If you skip this, nothing will work.
5. Click "Install Now".

### 2. Install Git (The Code Manager)
If your code is on GitHub, you need Git to download it.
1. Go to [git-scm.com/download/win](https://git-scm.com/download/win).
2. Download the **64-bit Git for Windows Setup**.
3. Run the installer. You can just click "Next" through all the default options.

### 3. Install VS Code (The Editor)
1. Go to [code.visualstudio.com](https://code.visualstudio.com/).
2. Download and install it. 
3. Open VS Code, go to Extensions (Ctrl+Shift+X), and install the **"Python"** extension by Microsoft.

### 4. NVIDIA GPU Setup (ONLY if the PC has an NVIDIA graphics card)
If the new PC has an NVIDIA GPU (like an RTX 3060, 4090, etc.), you must install CUDA so the AI can use the graphics card. If you skip this, the AI will use the CPU, which is 10x to 50x slower.
1. First, update your normal graphics drivers using GeForce Experience or the NVIDIA website.
2. Go to the [CUDA Toolkit 12.1 Archive](https://developer.nvidia.com/cuda-12-1-0-download-archive). *(PyTorch officially supports 12.1 currently).*
3. Select Windows -> x86_64 -> Windows 11 (or 10) -> exe (local).
4. Download the massive file (about 3GB) and install it using the "Express" installation.

---

## Phase 2: Getting the Project Files onto the New PC

You have two ways to move the `llm-project` to the new computer.

### Option A: Using a USB Drive or Zip File (Manual Transfer)
If you copy the folder manually, **do not copy everything**.
1. Copy the `llm-project` folder to your USB drive.
2. **IMPORTANT:** Delete the `.venv` folder from the USB drive. Virtual environments are tied to the exact Windows username and file paths of the old computer. If you copy it, it will break on the new PC.
3. Paste the folder onto the new PC (e.g., `C:\Users\YourName\Documents\projects\llm-project`).

### Option B: Using Git (If hosted online)
1. Open PowerShell on the new PC.
2. Type: `cd C:\Users\YourName\Documents`
3. Type: `git clone <your-github-url-here>`

---

## Phase 3: Setting up the Virtual Environment

A virtual environment is an isolated sandbox. It ensures the AI libraries for this project don't interfere with anything else on the computer.

1. Open **PowerShell** as an Administrator.
2. We need to allow PowerShell to run scripts. Type this exact command and hit Enter:
   ```powershell
   Set-ExecutionPolicy Unrestricted -Scope CurrentUser
   ```
   *(Press 'Y' if it asks for confirmation).*
3. Now, navigate to your project folder:
   ```powershell
   cd C:\Users\YourName\Documents\projects\llm-project
   ```
4. Create the new sandbox (this creates a new `.venv` folder):
   ```powershell
   python -m venv .venv
   ```

---

## Phase 4: Installing the AI Libraries (Dependencies)

Now we install PyTorch, Transformers, FastAPI, and everything else inside the sandbox.

1. Make sure you are still in the `llm-project` folder in PowerShell.
2. Run this command to install everything from your `requirements.txt`:
   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. **Wait.** This step will download several gigabytes of data (PyTorch alone is ~2.5GB). It might take 10-20 minutes depending on internet speed.

### Verifying the GPU Installation
Once the installation is done, let's verify that Python can see your graphics card. Run this command:
```powershell
.venv\Scripts\python.exe -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
- If it prints `GPU Available: True` and your graphics card name, **you are perfect.**
- If it prints `GPU Available: False`, your PC will run very slowly on the CPU. (Make sure you installed CUDA and update your drivers).

---

## Phase 5: Running the AI Pipeline

Everything is installed! You are ready to train the model. 

### The "One-Click" Method
You can run the entire process automatically using the batch file:
```powershell
.\run_pipeline.bat
```

### The Step-by-Step Method (Highly Recommended for First Time)
If you want to make sure each step works perfectly without crashing, run them one by one.

**Step 1: Scrape Data from the Web**
```powershell
.venv\Scripts\python.exe -m src.data.scraper
```
*(This creates `data/raw/scraped_data.txt`)*

**Step 2: Train the Tokenizer (The AI's Dictionary)**
```powershell
.venv\Scripts\python.exe -m src.data.train_tokenizer --vocab-size 32000
```
*(This creates `data/tokenizer.json`)*

**Step 3: Prepare the Training Data (Binary conversion)**
```powershell
# To use your local scraped data:
.venv\Scripts\python.exe -m src.data.prepare --dataset local

# OR to use real internet data (38 GB OpenWebText):
.venv\Scripts\python.exe -m src.data.prepare --dataset openwebtext
```
*(This creates `data/processed/train.bin` and `val.bin`)*

**Step 4: Pretrain the Model (The heavy lifting)**
```powershell
.venv\Scripts\python.exe -m src.training.train --set init_from=scratch max_iters=20000
```
*(This creates `out/ckpt.pt`. This step can take hours or days depending on your GPU).*

**Step 5: Fine-tune the Model (Make it a Chatbot)**
```powershell
.venv\Scripts\python.exe -m src.training.finetune
```
*(This creates `out/finetuned/ckpt.pt`)*

**Step 6: Talk to Your AI**
```powershell
# Open the chat interface in the terminal
.venv\Scripts\python.exe -m src.inference.chat --checkpoint out/finetuned/ckpt.pt

# OR Start the Web API Server (for Next.js)
.venv\Scripts\python.exe -m src.serving.api
```

---

## Common Errors & Fixes

1. **"python is not recognized as an internal or external command"**
   - **Cause:** You missed Step 4 in Phase 1.
   - **Fix:** Re-run the Python installer, choose "Modify", and check the "Add Python to environment variables" box.

2. **"CUDA out of memory" or "RuntimeError: CUDA error" during Training**
   - **Cause:** Your GPU doesn't have enough VRAM (Video RAM) to handle the batch size.
   - **Fix:** Open `config/default.yaml` and change `batch_size: 12` to `batch_size: 4` or `batch_size: 2`. You can also lower `block_size` to `128`.

3. **"No module named 'fastapi'"**
   - **Cause:** The dependencies didn't install correctly in the virtual environment.
   - **Fix:** Run `.venv\Scripts\python.exe -m pip install -r requirements.txt` again.

4. **"ModuleNotFoundError: No module named 'src'"**
   - **Cause:** You are running the command from the wrong folder.
   - **Fix:** Make sure you are inside the `llm-project` folder in PowerShell, not just the `projects` folder. Your path should look like `PS C:\Users\Name\...\llm-project>`.