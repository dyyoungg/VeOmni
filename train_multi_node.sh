

export WANDB_API_KEY="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" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"
export VEOMNI_USE_LIGER_KERNEL="1"
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
    /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/tasks/train_llavaomni.py \
    /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0808_audio_zh_all/14B_qwen25encoder_fsdp2_audiodown8.yaml