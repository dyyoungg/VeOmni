

export WANDB_API_KEY="" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"
export VIT_PROFILE="0"

cd /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/
echo "=== Distributed Training Environment Variables ==="
echo "MASTER_ADDR:  $MASTER_ADDR"
echo "MASTER_PORT:  $MASTER_PORT"
echo "=================================================="
nnodes=$1
nproc_per_node=$2

# fsdp2 + ulysses
torchrun \
    --nnodes $nnodes \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $nproc_per_node \
    tasks/train_llavaomni.py \
    test.yaml