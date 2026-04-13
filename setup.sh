set -e

echo ""
echo "============================================"
echo "   Text Summarizer — Setup"
echo "============================================"
echo ""

# ── Find a native ARM64 Python ───────────
ARM_PYTHON=""

CANDIDATES=(
  "/opt/homebrew/bin/python3.11"
  "/opt/homebrew/bin/python3.12"
  "/opt/homebrew/bin/python3.10"
  "/opt/homebrew/bin/python3"
)

for candidate in "${CANDIDATES[@]}"; do
  if [ -x "$candidate" ]; then
    ARCH=$("$candidate" -c "import platform; print(platform.machine())" 2>/dev/null || echo "")
    if [ "$ARCH" = "arm64" ]; then
      ARM_PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$ARM_PYTHON" ]; then
  echo "❌  No ARM64 (Apple Silicon) Python found."
  echo ""
  echo "  Install Homebrew and Python 3.11 first:"
  echo ""
  echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  echo "  brew install python@3.11"
  echo ""
  echo "  Then re-run:  bash setup.sh"
  echo ""
  exit 1
fi

echo "✅  Found Python  : $ARM_PYTHON"
echo "    Architecture  : $($ARM_PYTHON -c 'import platform; print(platform.machine())')"
echo "    Version       : $($ARM_PYTHON --version)"
echo ""

# ── Remove any old .venv ─────────────────
if [ -d ".venv" ]; then
  echo "🗑   Removing old .venv ..."
  rm -rf .venv
fi

# ── Create fresh venv ────────────────────
echo "📦  Creating virtual environment ..."
"$ARM_PYTHON" -m venv .venv
source .venv/bin/activate

VENV_ARCH=$(python3 -c "import platform; print(platform.machine())")
echo "    venv Python   : $(which python3)"
echo "    venv arch     : $VENV_ARCH"

if [ "$VENV_ARCH" != "arm64" ]; then
  echo ""
  echo "❌  venv is still $VENV_ARCH — something is wrong."
  echo "    Make sure Homebrew is installed at /opt/homebrew (Apple Silicon path)."
  exit 1
fi
echo ""

# ── Install packages ─────────────────────
echo "⬇️   Upgrading pip ..."
pip install --upgrade pip --quiet

echo "⬇️   Installing PyTorch ..."
pip install torch torchvision torchaudio --quiet

echo "⬇️   Installing project dependencies ..."
pip install -r requirements.txt --quiet

echo ""
echo "============================================"
echo "  ✅  Setup complete!"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Add dataset CSVs to  ML/data/"
echo "       samsum-train.csv"
echo "       samsum-validation.csv"
echo ""
echo "  2. Open and run the notebook:"
echo "       ML/text_summarizer.ipynb"
echo "     (this trains and saves the model)"
echo ""
echo "  3. Start the server:"
echo "       bash run.sh"
echo ""
echo "  4. Open in browser:"
echo "       http://localhost:8000"
echo "============================================"