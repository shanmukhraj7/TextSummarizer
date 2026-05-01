from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
import logging
import os

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(
    title="Text Summarizer",
    description="Dialogue summarization using a fine-tuned BART-base model on SAMSum.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()
logger.info(f"Running on device: {device}")


# ─────────────────────────────────────────────
#  Load model & tokenizer
# ─────────────────────────────────────────────
MODEL_PATH    = "ML/saved_summary_model"
FALLBACK_MODEL = "facebook/bart-base"

try:
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading local fine-tuned model from: {MODEL_PATH}")
    else:
        logger.info(f"Local model not found. Downloading '{FALLBACK_MODEL}' dynamically...")

    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL)

    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

except Exception as exc:
    logger.error(f"Failed to load model: {exc}")
    raise RuntimeError(f"Could not load the model. {exc}") from exc


# ─────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────
class DialogueInput(BaseModel):
    dialogue: str

    @field_validator("dialogue")
    @classmethod
    def must_have_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Dialogue must not be empty.")
        if len(v.strip()) < 10:
            raise ValueError("Dialogue is too short to summarize.")
        return v


class SummaryOutput(BaseModel):
    summary: str
    input_word_count: int
    summary_word_count: int


# ─────────────────────────────────────────────
#  Text helpers
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalize line endings, collapse whitespace, strip HTML."""
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()          # ← no .lower() — BART is case-sensitive


def run_summarize(dialogue: str) -> str:
    """Tokenize → generate → decode."""
    cleaned = clean_text(dialogue)   # ← no "summarize: " prefix for BART

    inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=512,              # ← Reduced to 512 for faster processing
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,      # ← helps BART produce fuller summaries
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize", response_model=SummaryOutput)
async def summarize(payload: DialogueInput):
    try:
        result = run_summarize(payload.dialogue)
        return SummaryOutput(
            summary=result,
            input_word_count=len(payload.dialogue.split()),
            summary_word_count=len(result.split()),
        )
    except Exception as exc:
        logger.error(f"Summarization error: {exc}")
        raise HTTPException(status_code=500, detail="Summarization failed. Please try again.")


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}