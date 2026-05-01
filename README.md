---
title: Text Summarizer
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 Text Summarizer — Fine-tuned BART on SAMSum

A production-ready **dialogue summarization system** built with a fine-tuned **BART-base** model, served through a **FastAPI** backend and a polished web UI.

---

## 📁 Project Structure

```
Text-Summarizer/
│
├── ML/
│   ├── data/                        ← Place dataset CSVs here
│   │   ├── samsum-train.csv
│   │   └── samsum-validation.csv
│   │
│   ├── saved_summary_model/         ← Created after training
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model.safetensors
│   │   ├── special_tokens_map.json
│   │   ├── merges.txt
│   │   ├── vocab.json
│   │   └── tokenizer_config.json
│   │
│   ├── preprocess.py                ← Text cleaning utilities
│   ├── tokenizing.py                ← Tokenization (512 max length)
│   └── text_summarizer.ipynb        ← Full training notebook
│
├── templates/
│   └── index.html                   ← Frontend web UI
│
├── app.py                           ← FastAPI application
├── requirements.txt
├── setup.sh                         ← One-time environment setup
├── run.sh                           ← Start the server
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (Apple Silicon Mac)

### Step 1 — Clone the repo

```bash
git clone https://github.com/shanmukhraj7/TextSummarizer.git
cd TextSummarizer
```

### Step 2 — Run setup (only once)

```bash
bash setup.sh
```

This will automatically:
- Find the correct ARM64 (Apple Silicon) Python via Homebrew
- Delete any broken `.venv` and create a fresh one
- Install all dependencies

> If Homebrew is not installed:
> ```bash
> /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
> brew install python@3.11
> ```

### Step 3 — Add the dataset

Download the SAMSum dataset from HuggingFace:
https://huggingface.co/datasets/samsum

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

This fine-tunes **BART-base** and saves the model to `ML/saved_summary_model/`.

### Step 5 — Start the server

```bash
bash run.sh
```

Open **http://localhost:8000** in your browser.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Model | `facebook/bart-base` via HuggingFace Transformers |
| Fine-tuning | HuggingFace `Trainer` API |
| Dataset | SAMSum (messenger-style dialogues + summaries) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Validation | Pydantic v2 |
| Hardware | Auto-detects MPS / CUDA / CPU |

---

## 🔌 API Reference

### `POST /summarize`

**Request:**
```json
{
  "dialogue": "Alice: Did you watch the game?\nBob: Yes! The last minute goal was incredible.\nAlice: I know, I almost fell off my chair!"
}
```

**Response:**
```json
{
  "summary": "Alice and Bob discuss last night's game and the amazing last-minute goal.",
  "input_word_count": 28,
  "summary_word_count": 13
}
```

### `GET /health`

```json
{ "status": "ok", "device": "mps" }
```

### `GET /`

Serves the web UI.

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
| Inference | Beam search, 4 beams, no-repeat-ngram=3, length penalty=2.0 |

---

## 🍎 Apple Silicon Note

Always use `python3 -m uvicorn` (not just `uvicorn`) on Apple Silicon. The plain `uvicorn` command spawns subprocesses that may inherit the system Intel Python instead of the venv ARM Python, causing a `cffi` architecture mismatch error. The `run.sh` script handles this automatically.

---

## 🔮 Possible Improvements

- Evaluate with ROUGE score metrics
- Add streaming output via Server-Sent Events
- Add Docker support for deployment
- Deploy to Hugging Face Spaces

---

## 📜 License

MIT — free to use, modify, and distribute.

---

## 👤 Author

**Avunoori Shanmukha Raj** — ML Engineer  
[GitHub](https://github.com/shanmukhraj7) · [LinkedIn](https://www.linkedin.com/in/shanmukha7/)