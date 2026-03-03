# Setting Up Your Virtual Environment

## What is a virtual environment and why do we use it?

A virtual environment is an **isolated Python installation** inside your project folder (typically `.venv/`). It has its own copy of `python`, `pip`, and every package you install.

Why this matters:

- **No conflicts.** Different projects can depend on different (even incompatible) versions of the same library. A virtual environment keeps each project's dependencies separate.
- **Reproducibility.** Everyone working on the project installs exactly the packages listed in `requirements.txt` — nothing more, nothing less.
- **Clean system.** Your OS-level Python stays untouched. If something breaks, just delete `.venv` and recreate it.

In short: **always work inside a virtual environment** for this course.

---

You may use any workflow you prefer:

1. **Terminal (recommended first choice)**
2. **VS Code workflow**
3. **Google Colab workflow**

_NOTE: If you are on Windows, read the **Windows + WSL** section before starting._

_NOTE: Before creating a new environment, if an old `.venv` setup failed, remove it and start clean. To remove a virtual environment, delete the `.venv` folder._

I recommend the terminal workflow for simplicity and flexibility (You can use it in a bare server environment, on any OS, and it is the most direct way to learn how virtual environments work).
VS Code is also a good option if you want an integrated experience. Google Colab is available if you want a cloud-based environment without local setup, but it has limitations.

---

## 1) Terminal Workflow (Recommended)

This is the preferred workflow for the course.

From the `SER_Project/` directory, run:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# (Optional) install any other package you need
# pip install <package_name>
pip install jupyterlab

# Download dataset
python download_data.py
```

### Launch Jupyter Lab directly

```bash
# make sure the .venv is activated first
jupyter lab
```

If `jupyter lab` is not found, install it in the active environment:

```bash
pip install jupyterlab
```

To deactivate your virtual environment (.venv) later:

```bash
# Just use this command:
deactivate
```

When `.venv` is activated, your shell automatically shows it, however the definitive check is:

```shell
which -a python
which -a jupyter
```

and you should see the PATH for current `.venv`. For windows users you may use `where` instead of `which`.

---

## 2) VS Code Workflow

### Prerequisites

- Install the **Python** extension (`ms-python.python`).
- Install the **Jupyter** extension (`ms-toolsai.jupyter`).

### Create and select environment

1. Open the `SER_Project/` folder in VS Code.
2. Open Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Windows/Linux).
3. Run **Python: Create Environment**.
4. Choose **Venv** and select Python 3.10+.
5. Install from `requirements.txt` when prompted.
6. Confirm interpreter via **Python: Select Interpreter** and choose `.venv`.

VS Code often auto-detects `.venv` and handles interpreter selection automatically, but always verify the selected interpreter in the status bar.

### VS Code installer for Windows users

You **must** install VS Code using the **System Installer** (not the User Installer). The User Installer can cause PATH and permission issues.

### Note on Jupyter notebooks in VS Code

When you open a `.ipynb` notebook, VS Code should prompt you to select a kernel. Choose the kernel associated with your `.venv` environment (it should be listed as something like `Python 3 ('.venv': venv)`).

You can run Jupyter lab directly from VS Code terminal as well (see Section 1). However, if you have Jupyter extension installed, you can directly open notebooks and select the `.venv` kernel without needing to launch Jupyter Lab separately. Just make sure to select the correct interpreter (the one from `.venv`) in VS Code when you open a notebook. As a VS Code user, I assume you are familiar with how to select the Python interpreter and Jupyter kernel in VS Code, but if not, you can find those options in the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) under **Python: Select Interpreter** and **Jupyter: Select Kernel** or for creating environment: **Python: Create environment**.

---

## 3) Google Colab Workflow

Use this if you do not want to manage a local environment. Keep in mind this is a cloud runtime and not a full replacement for local project structure.

### Mount Drive

Mount Google Drive when you need files beyond your current notebook runtime (for example `pd.read_csv("a_csv_file.csv")`). Place files in a known path such as `data/file.csv`, then load from that mounted path.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Organize dataset in Drive

Recommended layout:

```text
MyDrive/
└── SER_Project/
  └── data/
    ├── Audio_Speech_Actors_01-24/
    └── Audio_Song_Actors_01-24/
```

### Set data path

```python
DATA_DIR = "/content/drive/MyDrive/SER_Project/data"
```

### Install missing packages in Colab

```python
!pip install librosa soundfile xgboost
```

---

## Troubleshooting: Notebook does not detect `.venv` kernel

Before troubleshooting, I have found many cases that resolve easily if you explictly run jupyter lab from .venv terminal (Install ipykernel if it's not installed). To do this:

1. Make sure `.venv` is activated in your terminal.
2. `.venv/bin/jupyter lab` (or `.\venv\Scripts\jupyter lab` on Windows).
3. Open the notebook from the Jupyter Lab interface and select the kernel associated with `.venv`.
   If this does not work, then step below helps to register the kernel explicitly.

If you open a `.ipynb` notebook in **Jupyter Lab** or **VS Code** and your `.venv` environment is not listed as a kernel, follow these steps:

### 1. Install `ipykernel` inside `.venv`

Make sure your virtual environment is activated, then run:

```bash
pip install ipykernel
```

### 2. Register the kernel explicitly

```bash
python -m ipykernel install --user --name=ser_project --display-name "Python (SER Project)"
jupyter kernelspec list
```

This registers your `.venv` Python as a Jupyter kernel that both Jupyter Lab and VS Code can discover.

### 3. Refresh / restart

- **Jupyter Lab:** Refresh the browser page, then open your notebook and select **Kernel → Change Kernel → Python (.venv)**.
- **VS Code:** Reload the window (`Cmd+Shift+P` / `Ctrl+Shift+P` → **Developer: Reload Window**), open your notebook, click the kernel picker in the top-right corner, and select **Python (.venv)** (or the `.venv` interpreter).

### 4. Verify

Run this in a notebook cell to confirm the kernel is using the correct Python:

```python
import sys
print(sys.executable)
```

The output should point to a path inside your `.venv` directory (e.g., `.venv/bin/python`).

> **Tip:** If you ever delete and recreate `.venv`, you will need to repeat steps 1–2 because the old kernel registration points to a path that no longer exists.

---

## Windows Users (Read This First)

I do **not** directly support native Windows development environments in this course.

### Option A — Install WSL (Recommended)

This is the easiest path. Once WSL is installed you can follow the **exact same Linux/macOS commands** in Sections 1–2 above.

1. Open **PowerShell as Administrator** and run:

   ```powershell
   wsl --install -d Ubuntu
   ```

2. Restart your machine when prompted.
3. Open **Windows Terminal** — you will now see an **Ubuntu** option in the drop-down menu. Select it.
4. Create your Linux username and password when asked.

You are now in a full Linux shell. Follow Section 1 (Terminal Workflow) exactly as written — `python3`, `source`, `pip`, etc. all work normally.

If you use VS Code, open the project folder **from within WSL** so that `.venv` and kernels are detected correctly (e.g., run `code .` from the Ubuntu terminal inside the project directory).

- NOTE: You may need to search on how to integrate WSL with VS Code if you have not used it before, but it is straightforward and well-documented by Microsoft.
- **Performance tip:** For best performance, keep your project files inside the Linux filesystem (e.g., `~/projects/SER_Project/`) rather than under `/mnt/c/...`. Accessing Windows drives from WSL is noticeably slower.

---

### Option B — Native Windows Terminal / PowerShell

If you prefer not to install WSL and want to use the default Windows terminal or PowerShell, read **all** of the notes below carefully.

#### Activating the virtual environment

Create the virtual environment the same way:

```powershell
python -m venv .venv
```

Activation differs from Linux/macOS:

| Shell          | Command                      |
| -------------- | ---------------------------- |
| Command Prompt | `.venv\Scripts\activate`     |
| PowerShell     | `.venv\Scripts\Activate.ps1` |

> **Note:** On some Windows configurations the virtual-environment directory layout places binaries in `.venv\bin\` instead of `.venv\Scripts\`. If `.venv\Scripts\` does not exist, try `.venv\bin\activate` (or `.\bin\Activate.ps1`) instead. You can also check the contents of the `.venv` folder to see where the `activate` script is located.

#### PowerShell execution policy

PowerShell may block activation scripts by default. Allow them with:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

> **Important Note (PATH and Python toolchain issues):** Windows hides PATH configuration in various places (System Environment Variables, User Environment Variables, Windows Store Python aliases, etc.). If commands like `python`, `pip`, `urllib`, or other parts of the Python toolchain are not found — or if VS Code / the terminal does not reliably detect them — **it is your responsibility to locate and fix the PATH entries on your system.** Common things to check:
>
> - Make sure the **Windows Store Python app alias** is disabled (Settings → Apps → App execution aliases → turn off the `python.exe` / `python3.exe` entries).
> - Verify that your Python installation directory (e.g., `C:\Users\<you>\AppData\Local\Programs\Python\Python3xx\` and its `Scripts\` subfolder) appears in your **User** or **System** `PATH`.
> - After editing PATH, **restart** your terminal and VS Code for changes to take effect.
> - VS Code and terminal shells sometimes cache old PATH values; restarting the application fully (not just the terminal tab) can resolve detection issues with `pip`, `python`, `urlopen`, etc.

#### Run Jupyter lab or run python script:

Assuming you have successfully activated the virtual environment, you can proceed with installing dependencies and running the project as normal. Just be aware that Windows may require additional troubleshooting for PATH and execution policies, which is why I recommend using WSL for a smoother experience.

```powershell
# .venv is activated at this point

# Install dependencies for SER project
pip install -r requirements.txt

# install jupyterlab if you want to run it
pip install jupyterlab

# Run Jupyter Lab
jupyter lab

```

At this point, the launched jupyter lab should detect the `.venv` environment as a kernel option. If it does not, you may need to install the `ipykernel` package in your virtual environment:

```powershell
pip install ipykernel
```
