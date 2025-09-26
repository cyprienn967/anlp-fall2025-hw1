#!/usr/bin/env bash
set -euo pipefail

# (Optional) show GPU
nvidia-smi || true

# Use cu121 wheels that exist for Python 3.12
python -m pip install --upgrade pip

# PyTorch matching versions for 2.5.1 stack (works on Colab A100, CUDA 12.1 runtime wheels)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Other deps:
# - importlib-metadata is not needed on Py3.12; drop it to avoid conflicts.
# - tokenizers>=0.15 provides wheels for Py3.12; 0.13.3 will fail unless you install Rust and build.
pip install \
  tqdm==4.66.1 \
  requests==2.31.0 \
  filelock==3.0.12 \
  scikit-learn==1.2.2 \
  numpy==1.26.3 \
  tokenizers==0.15.2 \
  sentencepiece==0.1.99

# Model file
wget -q https://www.cs.cmu.edu/~vijayv/stories42M.pt -O stories42M.pt

# Sanity check
python - <<'PY'
import torch, torchvision, torchaudio, sys
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda?", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
PY
