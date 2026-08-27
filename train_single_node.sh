

export WANDB_API_KEY="" # replace with your API
export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"

cd /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev
# fsdp2
torchrun --nproc_per_node 8 tasks/train_llavaomni.py \
   test.yaml
