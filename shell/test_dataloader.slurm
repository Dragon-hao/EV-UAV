#!/bin/bash
#SBATCH --job-name=evuav_loader
#SBATCH --partition=g_l40
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dataloader-%j.out
#SBATCH --error=logs/dataloader-%j.err

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ev-uav

export CUDA_HOME=/fs/data/software/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/projects/EV-UAV/lib/hais_ops:$HOME/projects/EV-UAV:${PYTHONPATH:-}"

cd ~/projects/EV-UAV
mkdir -p logs

hostname
nvidia-smi

python - <<'PY'
import torch

from configs.configs import cfg
from dataset.ev_uav import EvUAV

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

dataset = EvUAV(cfg, mode="test")

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.custom_collate,
)

batch = next(iter(loader))

print("batch type:", type(batch))

for key, value in batch.items():
    print(
        key,
        type(value),
        getattr(value, "shape", None),
        getattr(value, "dtype", None),
        getattr(value, "device", None),
    )

print("DataLoader test passed")
PY
EOF