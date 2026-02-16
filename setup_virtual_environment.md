# Setting Up Your Virtual Environment

This guide covers three ways to set up your Python environment for the SER project.

---

## Option 1 — Terminal (macOS / Linux)

Open a terminal, navigate to the project folder, and run:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt

# Download the RAVDESS dataset
python download_data.py
```

To deactivate later, simply run `deactivate`.

### Windows Notes

On Windows the activation command differs (e.g. `.venv\Scripts\activate` in Command Prompt, or `.venv\Scripts\Activate.ps1` in PowerShell). You may also need to:

- Allow script execution in PowerShell (`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)
- Check that `python` or `py` is on your PATH

If you're using WSL, follow the macOS/Linux instructions above instead.

There are many environment-specific factors on Windows (antivirus, path conflicts, corporate policies, etc.) — you will need to troubleshoot for your own setup. The core steps (`python -m venv .venv` → activate → `pip install`) remain the same.

---

## Option 2 — VS Code

### Prerequisites

Install the **Python** extension (by Microsoft) from the Extensions marketplace (`ms-python.python`). This provides virtual environment management and Jupyter support.

### Create the environment

1. Open the `SER_Project/` folder in VS Code.
2. Open the Command Palette (`Cmd+Shift+P` on Mac, `Ctrl+Shift+P` on Windows/Linux).
3. Type **"Python: Create Environment"** and select it.
4. Choose **Venv**.
5. Select your preferred Python interpreter (e.g. Python 3.10+).
6. [optional] Check the box to install from `requirements.txt` when prompted.
   1. later you can install packages using `pip install numpy pandas matplotlib ...`.
7. VS Code will create a `.venv` folder and install the dependencies for you.

### Select the environment

After creation, make sure the bottom-left status bar shows the `.venv` interpreter. If it doesn't:

1. Open the Command Palette again.
2. Type **"Python: Select Interpreter"**.
3. Pick the `.venv` entry from the list.

### Jupyter kernel note

When you open a `.ipynb` notebook for the first time, VS Code may prompt you to install `ipykernel`. Accept the prompt — it is required to run notebook cells with your virtual environment. If you're not prompted, you can install it manually:

```bash
pip install ipykernel
```
    note this pip is activated environment pip

---

## Option 3 — Google Colab

If you prefer to work in Google Colab, the runtime already has most packages pre-installed. The main extra step is getting the **dataset** into the Colab environment.

### 1. Mount Google Drive

Run this cell at the top of your notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Follow the authorization prompt and give full access.

### 2. Upload / organize the data

Upload the RAVDESS data into your Google Drive so it is accessible after mounting. A recommended layout:

```
My Drive/
└── SER_Project/
    └── data/
        ├── Audio_Speech_Actors_01-24/
        └── Audio_Song_Actors_01-24/
```

### 3. Set the data path

Point your notebook to the mounted data directory:

```python
DATA_DIR = "/content/drive/MyDrive/SER_Project/data"
```

Use this `DATA_DIR` variable wherever your code references the `data/` folder. You can also directly use the path `/cocntent/drive/MyDrive/SER_Project_data`.

### 4. Install any missing packages

If a package is not available in the Colab runtime, install it inline:

```python
!pip install librosa soundfile xgboost
```

### 5. Upload project files (alternative to Drive)

For smaller files (scripts, the `minilearn/` package), you can also upload directly:

```python
from google.colab import files
uploaded = files.upload()  # opens a file picker
```

However, for the audio dataset (~400 MB), Google Drive mounting is strongly recommended over direct upload.
