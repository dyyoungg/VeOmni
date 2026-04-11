

export WANDB_API_KEY="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" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"

cd /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/
# fsdp2 + ulysses
torchrun --nproc_per_node 2 /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/tasks/train_llavaomni.py \
    /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/configs/multimodal/llavaomni/dense14B_ulysess.yaml \
    2>&1 | tee /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/train_ulysess.log

# # # dense fsdp2
# torchrun --nproc_per_node 4 /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/tasks/train_llavaomni.py \
#     /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/configs/multimodal/llavaomni/dense14B.yaml \
#     2>&1 | tee /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/train.log

# moe ep + fsdp2
# torchrun --nproc_per_node 8 /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/tasks/train_llavaomni.py \
#     /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/configs/multimodal/llavaomni/moe_fsdp2_ep.yaml \
#     2>&1 | tee /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/train_moe.log