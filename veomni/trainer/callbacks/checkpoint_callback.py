# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, List
import shutil

import torch
import torch.distributed as dist

from veomni.checkpoint import CheckpointerBase, build_checkpointer
from veomni.models import save_model_assets
from veomni.utils import helper
from veomni.utils.save_safetensor_utils import save_hf_safetensor, save_lora_adapter_with_dcp
from veomni.trainer.callbacks.base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


class CheckpointerCallback(Callback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1
        self.trainer.checkpointer: CheckpointerBase = build_checkpointer(
            dist_backend=args.train.accelerator.fsdp_config.fsdp_mode, ckpt_manager=args.train.checkpoint.manager
        )
        self.save_total_limit = getattr(args.train.checkpoint, "save_total_limit", None)

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)
            self._cleanup_old_checkpoints(state.global_step)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        # if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
        self._save_checkpoint(state)
        self._cleanup_old_checkpoints(state.global_step)


    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._load_checkpoint()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        """Block until the last async save finishes before tearing down the process group."""
        checkpointer = self.trainer.checkpointer
        if getattr(checkpointer, "save_future", None) is not None:  # async save
            logger.info_rank0("Waiting for the final async checkpoint save to finish...")
            checkpointer.save_future.result()
            checkpointer.save_future = None
            dist.barrier()
            logger.info_rank0("Final async checkpoint save finished.")

    def _load_checkpoint(self):
        """Load checkpoint from path."""
        args: "VeOmniArguments" = self.trainer.args
        if args.train.checkpoint.load_path is None:
            return

        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }

        self.trainer.checkpointer.wait_for_pending_save()

        self.trainer.checkpointer.load(
            args.train.checkpoint.load_path,
            state,
            trainable_only=bool(getattr(args.model, "lora_config", None)),
            parallel_state=self.parallel_state,
        )

        extra = state["extra_state"]
        self.trainer.state.global_step = extra["global_step"]
        self.trainer.start_epoch       = extra["start_epoch"]   # 直接用，不做计算
        self.trainer.start_step        = extra["start_step"]    # 直接用，不做计算

        self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])

        channel_loss_state = state["extra_state"].get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        # dataloader may only init on sp_rank_0 to save memory
        if (
            self.trainer.train_dataloader is not None
            and state["extra_state"].get("train_dataloader", None) is not None
        ):
            self.trainer.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        # Free transient buffers from DCP materialization before the first train step.
        # Large MoE resumes are often near GPU capacity; leftover allocator fragments
        # after load can OOM the first NCCL collective (e.g. grad-norm all-reduce).
        helper.empty_cache()

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "VeOmniArguments" = self.trainer.args

        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")

        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        channel_loss_state = channel_loss_callback.state_dict() if channel_loss_callback is not None else {}

        ckpt_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {
                "global_step": state.global_step,
                "start_epoch":  self.trainer.current_epoch,   # 当前是第几个 epoch
                "start_step":   self.trainer.current_step,    # 当前 epoch 内跑完了第几步
                "train_dataloader": self.trainer.train_dataloader.state_dict(),
                "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
                "environ_meter": self.trainer.environ_meter.state_dict(),
                "channel_loss_callback": channel_loss_state,
                "torch_rng_state": torch.get_rng_state(),
            },
        }

        # Free the training step's residual activations / autograd buffers
        # before DCP allocates NCCL collective buffers for the gather.
        # Mirrors the existing post-save ``empty_cache()`` below; without
        # this pre-save call the save can fight the training step for HBM
        # (observed as ``NCCL WARN Cuda failure 2 'out of memory'`` inside
        # dcp.save on Qwen3.5-35B-a3b VL h100x16). Cost: one ``cudaFree``
        # per ``save_steps``, well below noise.
        helper.empty_cache()

        self.trainer.checkpointer.save(
            save_checkpoint_path,
            ckpt_state,
            save_async=args.train.checkpoint.save_async,
            trainable_only=bool(getattr(args.model, "lora_config", None)),
            save_to_lowest_rank=args.train.checkpoint.dcp_save_to_lowest_rank,
            parallel_state=self.parallel_state,
        )

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


    def _get_saved_checkpoint_steps(self) -> List[int]:
        """按 global_step 升序返回已保存的所有 checkpoint 的 step 列表。"""
        args: "VeOmniArguments" = self.trainer.args
        save_dir = args.train.checkpoint.save_path
        print()
        if not os.path.isdir(save_dir):
            return []

        steps = []
        for name in os.listdir(save_dir):
            if name.startswith("global_step_"):
                try:
                    steps.append(int(name.split("_")[-1]))
                except ValueError:
                    pass
        return sorted(steps)

    def _cleanup_old_checkpoints(self, current_step: int) -> None:
        """保留最新的 save_total_limit 个 checkpoint，删除多余的旧 checkpoint。"""
        if not self.save_total_limit or self.save_total_limit <= 0:
            return

        if self.trainer.args.train.global_rank != 0:
            dist.barrier()
            return

        args: "VeOmniArguments" = self.trainer.args
        save_dir = args.train.checkpoint.save_path
        steps = self._get_saved_checkpoint_steps()

       
        if current_step not in steps:
            steps.append(current_step)
            steps.sort()

        to_delete = steps[: max(0, len(steps) - self.save_total_limit)]
        for step in to_delete:
            ckpt_dir = os.path.join(save_dir, f"global_step_{step}")
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
                logger.info_rank0(f"Deleted old checkpoint: {ckpt_dir}")

        dist.barrier()  


class HuggingfaceCkptCallback(CheckpointerCallback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.save_hf_weights = args.train.checkpoint.save_hf_weights
        self.every_n_steps = args.train.checkpoint.hf_save_steps
        self.every_n_epochs = args.train.checkpoint.hf_save_epochs

    def on_train_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state, stage="train_end")
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        # self._save_model_assets()
        super().on_train_begin(state)

    def _save_model_assets(self):
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0:
            save_model_assets(args.train.checkpoint.model_assets_dir, self.trainer.model_assets)
        dist.barrier()

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        """Save model in HuggingFace format."""
        args: "VeOmniArguments" = self.trainer.args
        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        dist.barrier()   # 先同步，所有 rank 一起到达
        if not os.path.exists(save_checkpoint_path):
            super()._save_checkpoint(state)

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()
            dist.barrier()

        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        logger.info_rank0(f"Saving HF weights to {hf_weights_path} ...")
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            model_assets=self.trainer.model_assets,
            ckpt_manager=args.train.checkpoint.manager,
            output_dir=args.train.checkpoint.output_dir,
            save_checkpoint_path=save_checkpoint_path,
            model=self.trainer.model,
            fqn_to_index_mapping=args.model.fqn_to_index_mapping,
            is_rank_0=args.train.global_rank == 0,
            parallel_state=self.parallel_state,
        )

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()
        logger.info_rank0(f"HF checkpoint saved at {hf_weights_path}")


class HFLoraCkptCallback(HuggingfaceCkptCallback):
    """Save LoRA HF weights alongside the DCP checkpoint."""

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        args: "VeOmniArguments" = self.trainer.args
        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")

        dist.barrier()
        if not os.path.exists(save_checkpoint_path):
            CheckpointerCallback._save_checkpoint(self, state)

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()
            dist.barrier()

        if stage == "train_end":
            self.trainer.optimizer = None
            self.trainer.lr_scheduler = None

        lora_save_path = os.path.join(args.train.checkpoint.output_dir, f"global_step_{state.global_step}")
        logger.info_rank0(f"Saving LoRA adapter to {lora_save_path} ...")
        save_lora_adapter_with_dcp(
            model=self.trainer.model,
            save_path=lora_save_path,
            adapter_name="default",
        )

        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step
        logger.info_rank0(f"LoRA checkpoint saved at {lora_save_path}")
