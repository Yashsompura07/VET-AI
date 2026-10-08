#!/usr/bin/env bash
# VetAI - one-command runner (Mac / Linux)
cd "$(dirname "$0")/backend"

echo "============================================"
echo "   VetAI - starting up"
echo "============================================"

# 1. Check python
if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] Python 3 not found. Install Python 3.10-3.13 from https://www.python.org/downloads/"
  exit 1
fi
echo "Using $(python3 --version)"

# 2. venv
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate

# 3. core deps (required)
echo "Installing core dependencies (first run takes a few minutes)..."
pip install --quiet --upgrade pip
if ! pip install --quiet -r requirements.txt; then
  echo "[ERROR] Could not install core dependencies."; exit 1
fi

# 4. photo deps (optional, non-fatal)
if [ -f "app/ml/image_model.keras" ]; then
  echo "Installing photo-detection support (TensorFlow, large)..."
  if ! pip install --quiet -r requirements-image.txt; then
    echo "[NOTE] TensorFlow did not install - VetAI will run with SYMPTOMS ONLY."
  fi
fi

# 5. train symptom model if needed
if [ ! -f "app/ml/symptom_model.joblib" ]; then
  echo "Preparing the disease model (one-time, ~1 minute)..."
  python train_symptom_model.py || { echo "[ERROR] Model preparation failed."; exit 1; }
fi

echo ""
echo "============================================"
echo "   VetAI is running."
echo "   Open in your browser:  http://127.0.0.1:8000"
echo "   Press CTRL+C to stop."
echo "============================================"
echo ""
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
