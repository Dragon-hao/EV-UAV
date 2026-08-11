#!/bin/bash
#SBATCH --job-name=evuav_test
#SBATCH --partition=g_l40
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/test-%j.out
#SBATCH --error=logs/test-%j.err

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ev-uav

export CUDA_HOME=/fs/data/software/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/projects/EV-UAV/lib/hais_ops:$HOME/projects/EV-UAV:${PYTHONPATH:-}"

cd ~/projects/EV-UAV
mkdir -p logs

echo "===== Node ====="
hostname
nvidia-smi

echo "===== Test ====="
python test.py
