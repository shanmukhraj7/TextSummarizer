---
title: Text Summarizer
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 Text Summarizer — Fine-tuned BART on SAMSum

A production-ready **dialogue summarization system** built by fine-tuning **BART-base** on the SAMSum dataset. It exposes a **FastAPI** backend with a polished dark-mode web UI and is fully containerized with Docker for deployment on Hugging Face Spaces or any cloud platform.

---

## ✨ Features

- 🤖 **Fine-tuned BART-base** — trained on SAMSum messenger-style dialogues using HuggingFace `Trainer` API
- ⚡ **FastAPI backend** — async REST API with Pydantic v2 input validation and Jinja2 templating
- 🎨 **Polished Web UI** — dark-mode interface with grain overlay, smooth animations, and real-time word-count stats (Syne + DM Sans fonts)
- 🍎 **Apple Silicon native** — auto-detects MPS / CUDA / CPU; `setup.sh` enforces ARM64 Python to avoid architecture mismatches
- 🐳 **Docker-ready** — CPU-optimized image targeting port 7860 for Hugging Face Spaces deployment
- 🔄 **Graceful fallback** — if the local fine-tuned model is absent, the server dynamically downloads `facebook/bart-base` from HuggingFace

---

## 📁 Project Structure

```
TextSummarizer/
│
├── ML/
│   ├── data/                        ← Place dataset CSVs here (not tracked by git)
│   │   ├── samsum-train.csv
│   │   └── samsum-validation.csv
│   │
│   ├── saved_summary_model/         ← Created after training (not tracked by git)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── merges.txt
│   │   ├── vocab.json
│   │   └── tokenizer_config.json
│   │
│   ├── preprocess.py                ← Text cleaning utilities (strip HTML, normalize whitespace)
│   ├── tokenizing.py                ← BART tokenization (512 / 128 max tokens)
│   └── text_summarizer.ipynb        ← Full training notebook
│
├── templates/
│   └── index.html                   ← Frontend web UI (dark-mode, animated)
│
├── app.py                           ← FastAPI application (routes, model loading, inference)
├── requirements.txt                 ← Python dependencies
├── Dockerfile                       ← CPU-optimized Docker image (port 7860)
├── setup.sh                         ← One-time Apple Silicon environment setup
├── run.sh                           ← Start the local dev server (port 8000)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Local — Apple Silicon Mac)

### Step 1 — Clone the repo

```bash
git clone https://github.com/shanmukhraj7/TextSummarizer.git
cd TextSummarizer
```

### Step 2 — Run setup (once only)

```bash
bash setup.sh
```

This script will:
- Scan Homebrew for a native ARM64 Python (3.10 / 3.11 / 3.12)
- Wipe any stale `.venv` and create a fresh ARM64 virtual environment
- Upgrade pip, install PyTorch (with MPS support), and install all project dependencies

> **No Homebrew?** Install it first:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> brew install python@3.11
> ```

### Step 3 — Add the dataset

Download the SAMSum dataset from HuggingFace:
👉 https://huggingface.co/datasets/samsum

Place these two files inside `ML/data/`:
```
ML/data/samsum-train.csv
ML/data/samsum-validation.csv
```

### Step 4 — Train the model

Open and run all cells in:
```
ML/text_summarizer.ipynb
```

This fine-tunes **BART-base** on 4,000 training samples and saves the model to `ML/saved_summary_model/`.

### Step 5 — Start the server

```bash
bash run.sh
```

Open **http://localhost:8000** in your browser.

> `run.sh` uses `python3 -m uvicorn` (not the bare `uvicorn` binary) to ensure subprocesses inherit the ARM64 venv Python and avoid `cffi` architecture mismatches.

---

## 🐳 Docker / Hugging Face Spaces

The included `Dockerfile` builds a lightweight CPU image suited for free-tier deployments:

```bash
# Build
docker build -t text-summarizer .

# Run
docker run -p 7860:7860 text-summarizer
```

Open **http://localhost:7860**.

> PyTorch is pinned to the CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`) to keep the image size small.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Model | `facebook/bart-base` via HuggingFace Transformers |
| Fine-tuning | HuggingFace `Trainer` API |
| Dataset | SAMSum (messenger-style dialogues + summaries) |
| Backend | FastAPI 0.115 + Uvicorn 0.30 |
| Frontend | HTML / Vanilla CSS + JS (Syne & DM Sans fonts) |
| Templating | Jinja2 3.1 |
| Validation | Pydantic v2 |
| Hardware | Auto-detects MPS (Apple Silicon) → CUDA → CPU |
| Container | Docker (`python:3.11-slim`, port 7860) |

---

## 🔌 API Reference

### `GET /`
Serves the web UI.

---

### `POST /summarize`

Accepts a text body and returns a summary with word-count statistics.

**Request:**
```json
{
  "text": "Alice: Did you watch the game last night?\nBob: Yes! That last-minute goal was incredible.\nAlice: I know, I almost fell off my chair!"
}
```

**Response:**
```json
{
  "summary": "Alice and Bob discuss last night's game and the amazing last-minute goal.",
  "input_word_count": 30,
  "summary_word_count": 14
}
```

**Validation rules (Pydantic v2):**
- `text` must not be empty or blank
- `text` must be at least 10 characters long

**Error response (422):** returned when validation fails.
**Error response (500):** returned if the model inference fails.

---

### `GET /health`

```json
{ "status": "ok", "device": "mps" }
```

Returns the current compute device (`mps`, `cuda`, or `cpu`).

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| Base model | `facebook/bart-base` |
| Dataset | SAMSum |
| Train samples | 4,000 (randomly sampled from 14,732) |
| Validation samples | 500 (randomly sampled from 818) |
| Epochs | 4 |
| Batch size | 1 (with 4 gradient accumulation steps) |
| Max input tokens | 512 |
| Max target tokens | 128 |
| Inference | Beam search — 4 beams, `no_repeat_ngram_size=3`, `length_penalty=2.0` |
| Output length | min 30 tokens, max 200 tokens |

> BART does **not** use task prefixes (unlike T5). Inputs are passed directly without a `"summarize: "` prefix.

---

## 🔧 ML Utilities

### `ML/preprocess.py`
```python
clean_data(text: str) -> str
```
Strips HTML tags, normalizes line endings, collapses extra whitespace, and lowercases the text. Used during dataset preprocessing in the training notebook.

### `ML/tokenizing.py`
```python
tokenizing_raw_data(row: dict) -> dict
```
Tokenizes dialogue → input IDs (max 512 tokens) and summary → labels (max 128 tokens). Padding token IDs in labels are replaced with `-100` so they are ignored by the cross-entropy loss.

---

## 🍎 Apple Silicon Note

Always launch uvicorn as `python3 -m uvicorn app:app ...` rather than the bare `uvicorn` command. The bare binary can spawn a subprocess that inherits the system's Intel Python instead of the venv's ARM64 Python, causing a `cffi` architecture mismatch error. The `run.sh` script handles this automatically.

---

## 🔮 Possible Improvements

- Evaluate with ROUGE-1 / ROUGE-2 / ROUGE-L metrics
- Add streaming output via Server-Sent Events (SSE)
- Fine-tune a larger model (e.g., `facebook/bart-large-cnn`)
- Add support for article / news summarization (beyond dialogues)
- CI/CD pipeline with automated testing

---

## 📦 Dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
transformers==4.44.2
torch>=2.0.0
sentencepiece==0.2.0
pydantic==2.8.2
jinja2==3.1.4
pandas==2.2.2
```

---

## 📜 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

**Avunoori Shanmukha Raj** — ML Engineer
[GitHub](https://github.com/shanmukhraj7) · [LinkedIn](https://www.linkedin.com/in/shanmukha7/)
