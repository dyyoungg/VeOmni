

export WANDB_API_KEY="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" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"
export VIT_PROFILE="0"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22   
export NCCL_IB_RETRY_CNT=13 
export NCCL_IB_AR_THRESHOLD=0
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
# source /opt/dtk/env.sh
# source /opt/dtk/cuda/env.sh
# source /opt/MacCodecSDK/env.sh
# export LD_LIBRARY_PATH=/opt/dtk/cuda/cuda/lib64:/opt/MacCodecSDK/lib:${LD_LIBRARY_PATH:-}
# export HSA_FORCE_FINE_GRAIN_PCIE=1
# export GPU_MAX_HW_QUEUES=1
export AOSS_FILE="/mnt/afs/yangdeyu/aoss_ydy_game.conf"
export VEOMNI_USE_LIGER_KERNEL="1"
export VEOMNI_FLCE_BIG_CHUNK=1

cd /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev
echo "=== Distributed Training Environment Variables ==="
echo "MASTER_ADDR:  $MASTER_ADDR"
echo "MASTER_PORT:  $MASTER_PORT"
echo "=================================================="
nnodes=$1
nproc_per_node=$2
yaml_path=$3

# fsdp2
torchrun \
    --nnodes $nnodes \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $nproc_per_node \
    tasks/train_llavaomni.py \
    $yaml_path
