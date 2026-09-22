

#!/usr/bin/env bash
set -uo pipefail
export WANDB_API_KEY="" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/VeOmni:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VEOMNI_USE_LIGER_KERNEL=1
export VIT_PROFILE=0
export PYTORCH_MUSA_ALLOC_CONF="expandable_segments:True"

# K8s网络配置
export GLOO_SOCKET_IFNAME=eth0
export MCCL_SOCKET_IFNAME=net1
export MCCL_IB_HCA="mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
export MCCL_IB_GID_INDEX=3
export MCCL_IB_TIMEOUT=22
export MCCL_IB_RETRY_CNT=13
export MCCL_IB_AR_THRESHOLD=0
export MCCL_DEBUG=WARN

export TORCH_MUSA_FSDP2_ENABLE_CE_COMM=0
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=0

# 修复container-toolkit注入的libmusa通用入口
MUSA_DRIVER_LIB=/usr/lib/x86_64-linux-gnu/libmusa.so.4.3.5
test -s "${MUSA_DRIVER_LIB}"

mkdir -p /tmp/musa-driver-compat
ln -sfn "${MUSA_DRIVER_LIB}" /tmp/musa-driver-compat/libmusa.so.1
ln -sfn "${MUSA_DRIVER_LIB}" /tmp/musa-driver-compat/libmusa.so
export LD_LIBRARY_PATH="/tmp/musa-driver-compat:${LD_LIBRARY_PATH:-}"

cd /mnt/afs/yangdeyu/VeOmni
# python /mnt/afs/yangdeyu/VeOmni/test.py
# fsdp2
torchrun --nproc_per_node 8 tasks/train_llavaomni.py \
   /mnt/afs/yangdeyu/VeOmni/test.yaml
