#!/bin/bash
#SBATCH --job-name=evuav_forward
#SBATCH --partition=g_l40
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/forward-%j.out
#SBATCH --error=logs/forward-%j.err

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ev-uav

export CUDA_HOME=/fs/data/software/cuda/11.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$HOME/projects/EV-UAV/lib/hais_ops:$HOME/projects/EV-UAV:${PYTHONPATH:-}"

cd ~/projects/EV-UAV
mkdir -p logs

python - <<'PY'
import torch
from torch.utils.data import DataLoader

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet

print("GPU:", torch.cuda.get_device_name(0))

dataset = EvUAV(cfg, mode="test")

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=dataset.custom_collate,
)

batch = next(iter(loader))

print("batch keys:", batch.keys())
print("voxel features:", batch["voxel_ev"].features.shape)
print("voxel indices:", batch["voxel_ev"].indices.shape)

net = evspsegnet(cfg).cuda()
net.eval()

print("model parameters:", sum(p.numel() for p in net.parameters()))

with torch.no_grad():
    output = net(batch["voxel_ev"])

print("output type:", type(output))

if isinstance(output, dict):
    for key, value in output.items():
        print(
            key,
            type(value),
            getattr(value, "shape", None),
            getattr(value, "dtype", None),
            getattr(value, "device", None),
        )
elif isinstance(output, (tuple, list)):
    for i, value in enumerate(output):
        print(
            i,
            type(value),
            getattr(value, "shape", None),
            getattr(value, "dtype", None),
            getattr(value, "device", None),
        )
else:
    print(
        "output shape:", getattr(output, "shape", None),
        "dtype:", getattr(output, "dtype", None),
        "device:", getattr(output, "device", None),
    )

print("Forward test passed")
PY