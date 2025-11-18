Install WSL2 (not WSL)

## 1. Install system-level dependencies (inside WSL2)

Open your **WSL2 terminal** and run:

```bash
sudo apt update

# Core build + Python headers (needed for PyStan compilation)
sudo apt install -y \
  python3 python3-pip python3-venv python3-dev \
  build-essential g++ \
  git
```

> ⚠️ PyStan 2.19.x is happiest with Python ≤ 3.9.
> If your system Python is 3.10/3.11 and you hit issues, we can switch to a conda env with Python 3.9—but try this first; many people can still compile from source on Linux.

---

## 2. Create a project folder & virtual environment

Pick a folder where you want the notebook to live, e.g.:

```bash
mkdir -p ~/cuped_assertions
cd ~/cuped_assertions
```

Create & activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

---

## 3. Upgrade `pip` and basic Python tooling

```bash
pip install --upgrade pip wheel setuptools
```

---

## 4. Install Python packages you need

Core scientific stack + Jupyter + PyStan:

```bash
pip install \
  numpy \
  cython \
  pandas \
  matplotlib \
  seaborn \
  jupyterlab \
  ipykernel \
  pystan
```

Optional but often handy (if you reuse bits of the rubric notebook):

```bash
pip install statsmodels scipy
```

Quick sanity check:

```bash
python -c "import numpy, pandas, pystan; print('OK:', numpy.__version__, pandas.__version__)"
```

If that prints `OK: ...` without errors, you’re good.

---

## 5. Expose this venv as a Jupyter kernel

Still inside the activated venv (`(.venv)`):

```bash
python -m ipykernel install --user \
  --name cuped_assertions_env \
  --display-name "CUPED Assertions (PyStan)"
```

This registers a kernel called **“CUPED Assertions (PyStan)”** in Jupyter.

---

## 6. Start Jupyter (Notebook or Lab)

From the same folder, still in the venv:

```bash
jupyter lab
# or, if you prefer the classic UI:
# jupyter notebook
```

Jupyter will print a URL like `http://localhost:8888/?token=...`.

Because you’re on WSL2:

* Open that URL in your **Windows browser** (Chrome/Edge/etc.).
* If it doesn’t auto-open, just copy-paste from the terminal.

---

## 7. Create / open the notebook and attach the kernel

1. In the Jupyter UI:

   * Click **“New” → Notebook** (or in JupyterLab: `File → New Notebook`).
2. When asked to pick a kernel, choose:

   * **“CUPED Assertions (PyStan)”**
3. Paste the Bayesian CUPED notebook code I gave you into the cells (or open your saved `.ipynb` file).
4. Run cells top to bottom.

---

## 8. Next time you come back

From WSL2:

```bash
cd ~/cuped_assertions
source .venv/bin/activate
jupyter lab   # or jupyter notebook
```

Kernel will already be registered; just pick **“CUPED Assertions (PyStan)”**.