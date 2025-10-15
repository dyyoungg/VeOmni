export PYTHONPATH="/mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev:$PYTHONPATH"

torchrun --nnodes=1 --nproc-per-node=8 --master-port=4321 tests/utils/test_trainer_saveload.py \
    --model.model_path /mnt/afs/share/Qwen3-30B-A3B-Instruct-2507-veomni-merge \
    --model.moe_implementation fused \
    --model.attn_implementation flash_attention_2 \
    --train.expert_parallel_size 8 \
    --train.global_batch_size 8 \
    --train.micro_batch_size 1 \
    --data.max_seq_len 128 \
    --data.train_path "dummy" \
    --train.output_dir ./test_trainer_saveload \
    --train.max_steps 20 \
    --train.rmpad false \
    --train.rmpad_with_pos_ids true \
    --train.data_parallel_mode "fsdp2" \
    --train.init_device "meta" \
    --train.ckpt_manager "dcp" \
    --train.enable_gradient_checkpointing True \
    --train.enable_activation_offload True \
