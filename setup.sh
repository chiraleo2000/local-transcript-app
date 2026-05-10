#!/bin/bash
echo "============================================"
echo " Local Transcript App - Setup (Linux/Mac)"
echo "============================================"
echo

# Step 1: Create virtual environment
echo "[1/5] Creating virtual environment..."
python3 -m venv venv || { echo "ERROR: Failed to create venv."; exit 1; }

# Step 2: Activate virtual environment
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "[3/5] Upgrading pip..."
python -m pip install --upgrade pip

# Step 4: Install OpenVINO first (pinned version)
echo "[4/5] Installing OpenVINO 2026.1.0..."
pip install openvino==2026.1.0

# Step 5: Install remaining dependencies
echo "[5/5] Installing remaining dependencies..."
pip install -r requirements.txt

# torchcodec is incompatible on Windows without FFmpeg full-shared DLLs.
# Transformers 4.57+ tries to import it for ASR pipelines; remove it so
# transformers falls back to soundfile/librosa (which works correctly).
pip uninstall torchcodec -y 2>/dev/null || true

# Create a minimal torchcodec stub so transformers 4.57 metadata check
# does not crash with PackageNotFoundError after the real package is removed.
echo "[6/6] Creating torchcodec compatibility stub..."
python -c "
import os, sys
d = os.path.join(sys.prefix, 'lib', 'python' + sys.version[:3], 'site-packages', 'torchcodec-0.0.1.dist-info')
os.makedirs(d, exist_ok=True)
open(os.path.join(d,'METADATA'),'w').write('Metadata-Version: 2.1\nName: torchcodec\nVersion: 0.0.1\n')
open(os.path.join(d,'RECORD'),'w').write('')
open(os.path.join(d,'INSTALLER'),'w').write('pip')
print('torchcodec stub created.')
"

# Install pywebview so the launcher opens a native desktop window.
echo "[7/7] Installing pywebview (native desktop window)..."
pip install pywebview

echo
echo "============================================"
echo " Setup complete!"
echo
echo " To run the app:"
echo "   source venv/bin/activate"
echo "   ./run.sh"
echo
echo " Prepare/download local models:"
echo "   python scripts/bootstrap_models.py"
echo "============================================"
