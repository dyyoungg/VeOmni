

export WANDB_API_KEY="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" # replace with your API
export PYTHONPATH="/home/tione/notebook/yangdeyu/VeOmni:$PYTHONPATH"
# source /opt/dtk/env.sh
# source /opt/dtk/cuda/env.sh
# source /opt/MacCodecSDK/env.sh
# source /home/tione/notebook/yangdeyu/rccl_env.sh
# export LD_LIBRARY_PATH=/opt/dtk/cuda/cuda/lib64:/opt/MacCodecSDK/lib:${LD_LIBRARY_PATH:-}
# export HSA_FORCE_FINE_GRAIN_PCIE=1
# export GPU_MAX_HW_QUEUES=1
export AOSS_FILE="/home/tione/notebook/yangdeyu/aoss_ydy_game.conf"
export VEOMNI_USE_LIGER_KERNEL="1"
export VEOMNI_FLCE_BIG_CHUNK=1
export VEOMNI_FAST_RMS_NORM=1
export VEOMNI_COMPILE_FRIENDLY_CKPT=1

cd /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev
# fsdp2
torchrun --nproc_per_node 2 tasks/train_llavaomni.py \
   /mnt/afs/yangdeyu/GameMLLM/VeOmni-Dev/exp_data/0725_stage3_imagesft_removesubtitle_addvideoxl/30A3B_qwen35encoder_fsdp2_freeze_router_auxloss_dynamic_downsample.yaml
