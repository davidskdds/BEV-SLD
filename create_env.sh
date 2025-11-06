#!/usr/bin/env bash
set -e  # exit if any command fails

echo "[INFO] Installing python3-venv (requires sudo)..."
sudo apt-get update -y
sudo apt-get install -y python3-venv

echo "[INFO] Creating virtual environment: bev_sld_env"
python3 -m venv bev_sld_env

echo "[INFO] Activating environment"
# shellcheck disable=SC1091
source bev_sld_env/bin/activate

echo "[INFO] Upgrading pip"
python -m pip install --upgrade pip

echo "[INFO] Installing requirements"
pip install -r requirements.txt

echo ""
echo "✅ Virtual environment created successfully!"
echo "To activate it later, run:"
echo "source bev_sld_env/bin/activate"
