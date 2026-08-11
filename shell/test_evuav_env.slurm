#!/bin/bash
#SBATCH --job-name=evuav_env
#SBATCH --partition=g_l40
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_log/evuav_env.%j.out
#SBATCH --error=slurm_log/evuav_env.%j.err

set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ev-uav

echo "===== Node ====="
hostname
nvidia-smi

echo "===== Environment ====="
which python

python - <<'PY'
import numpy
import torch
import spconv.pytorch as spconv

print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

a = torch.randn(1024, 1024, device="cuda")
b = a @ a
print("PyTorch result:", b.mean().item())

indices = torch.tensor(
    [
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 2, 2, 2],
    ],
    dtype=torch.int32,
    device="cuda",
)

features = torch.randn(4, 4, device="cuda")

x = spconv.SparseConvTensor(
    features=features,
    indices=indices,
    spatial_shape=[4, 4, 4],
    batch_size=1,
)

conv = spconv.SubMConv3d(
    4,
    8,
    kernel_size=3,
    padding=1,
    bias=False,
).cuda()

with torch.no_grad():
    y = conv(x)

print("spconv output:", y.features.shape)
print("All GPU tests passed")
PY