# Development Setup (uv)

This project uses **uv** to manage the Python virtual environment and dependencies.

---

## Windows (PowerShell)

### 1. Install uv

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```


### 2. Create & sync the virtual environment

```powershell
uv sync --all-extras
```

This creates `.venv/` and installs all dependencies.

---

### 3. Activate the virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

---

### 4. Run commands normally

```powershell
python your_script.py
p2pfl your_command
pytest -v
```

No `uv run` is needed after activation.

---

## macOS / Linux (Terminal)

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

### 2. Create & sync the virtual environment

```bash
uv sync --all-extras
```

---

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

---

### 4. Run commands normally

```bash
python your_script.py
p2pfl your_command
pytest -v
```

No `uv run` is needed after activation.

---

## TensorBoard Logs (Lightning)

Training and evaluation metrics are logged automatically to the `lightning_logs/` directory by PyTorch Lightning.

### View logs

After running an experiment, start TensorBoard from the project root:

```bash
tensorboard --logdir lightning_logs
```

Open the URL shown in the terminal (usually):

```
http://localhost:6006
```

### What you will see

- Each `version_X/` corresponds to **one Lightning run** (not a node)
- Metrics such as training loss, validation loss, and accuracy appear under **Scalars**
- Multiple versions are normal when running multiple nodes or rounds

### Clean logs (optional)

To start fresh before a new run:

```bash
rm -rf lightning_logs
mkdir lightning_logs
```

(Use PowerShell equivalents on Windows if needed.)

---

## Notes

- Always run `uv sync --all-extras` after pulling new changes
- The virtual environment lives in `.venv/`
- Workflow is identical across platforms once activated

## Docker

You can also use the library with Docker. We provide a Docker image with the library installed. You can use it as follows:

```bash
docker build -t p2pfl .
docker run -it --rm p2pfl bash
```

## Create testing scenario

You can check how to create a scenario via p2pfl documentation or check mnist_random10.py at root
