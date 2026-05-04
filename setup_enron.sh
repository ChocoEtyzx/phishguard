#!/usr/bin/env bash
# ============================================================
# PhishGuard — Enron Dataset Setup Script
# Run this ONCE on your local machine to download the dataset.
# ============================================================

set -e

echo "=== PhishGuard: Enron Dataset Setup ==="

# 1. Install kaggle CLI if missing
if ! command -v kaggle &>/dev/null; then
    echo "Installing Kaggle CLI..."
    pip install kaggle
fi

# 2. Save token using the file method (forces the CLI to recognize it)
mkdir -p ~/.kaggle

# 3. Download the Enron spam dataset
echo "Downloading Enron spam dataset..."
kaggle datasets download -d wanderfj/enron-spam --unzip --quiet
echo "✅ Dataset downloaded!"

# 4. Check what CSV was extracted
echo ""
echo "Files in current directory:"
ls -lh *.csv 2>/dev/null || echo "No CSV found — check extracted folder names below:"
ls -la

# 5. Run training
echo ""
echo "=== Starting model training ==="
python train_model.py