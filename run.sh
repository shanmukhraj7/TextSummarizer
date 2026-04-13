#!/bin/bash
# ──────────────────────────────────────────────────
#  Text Summarizer — Start server
#  Usage:  bash run.sh
# ──────────────────────────────────────────────────

# ── Check .venv exists ────────────────────────────
if [ ! -d ".venv" ]; then
  echo ""
  echo "❌  .venv not found."
  echo "    Run setup first:  bash setup.sh"
  echo ""
  exit 1
fi

# ── Activate ─────────────────────────────────────
source .venv/bin/activate

# ── Verify architecture ───────────────────────────
ARCH=$(python3 -c "import platform; print(platform.machine())" 2>/dev/null)
if [ "$ARCH" != "arm64" ]; then
  echo ""
  echo "❌  Wrong Python architecture: $ARCH"
  echo "    Delete .venv and re-run:  bash setup.sh"
  echo ""
  exit 1
fi

# ── Check model exists ────────────────────────────
if [ ! -d "ML/saved_summary_model" ]; then
  echo ""
  echo "❌  Trained model not found at ML/saved_summary_model/"
  echo ""
  echo "    Train the model first:"
  echo "    1. Add CSVs to ML/data/"
  echo "    2. Run ML/text_summarizer.ipynb"
  echo ""
  exit 1
fi

echo ""
echo "============================================"
echo "  Text Summarizer"
echo "  http://localhost:8000"
echo "  Press CTRL+C to stop"
echo "============================================"
echo ""

# ── IMPORTANT: use python3 -m uvicorn, NOT just uvicorn
#    This forces subprocesses to use the venv Python,
#    preventing the x86 / arm64 architecture mismatch.
python3 -m uvicorn app:app --reload --host 0.0.0.0 --port 8000