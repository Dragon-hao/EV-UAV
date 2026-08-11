#!/bin/bash
#SBATCH --job-name=evuav_async
#SBATCH --partition=g_l40
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=logs/async-%j.out
#SBATCH --error=logs/async-%j.err


set -euo pipefail


# =========================
# Conda
# =========================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ev-uav


# =========================
# CUDA
# =========================

export CUDA_HOME=/fs/data/software/cuda/11.8

export PATH="$CUDA_HOME/bin:$PATH"


# 避免 libtinfo.so.6 warning
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"


# =========================
# Project
# =========================

export PYTHONPATH="$HOME/projects/EV-UAV/lib/hais_ops:$HOME/projects/EV-UAV:${PYTHONPATH:-}"


cd ~/projects/EV-UAV


mkdir -p logs


echo "===== Node ====="
hostname

echo "===== GPU ====="
nvidia-smi


echo "===== Environment ====="

which python

python - <<'PY'
import torch
import spconv.pytorch as spconv
import HAIS_OP

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("HAIS_OP:", HAIS_OP.__file__)
PY


echo "===== Async Evaluation ====="

python test_async_window.py 