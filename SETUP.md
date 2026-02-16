# New PC Setup Instructions 🚀

If you move this project to a new computer, follow these two steps to get it running:

### 1. Create the Virtual Environment
Open your terminal in the project folder and run:
```powershell
python -m venv .venv
```

### 2. Install all Dependencies
Once the environment is created, install the required libraries (torch, numpy, etc.) by running:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
---

### ⚠️ Note for Git LFS
If you cloned this from GitHub, don't forget to run this once to download the actual model data:
```powershell
git lfs pull
```
