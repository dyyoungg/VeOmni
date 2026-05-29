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
from veomni.utils.save_safetensor_utils import save_hf_safetensor
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
        self.trainer.checkpointer: CheckpointerBase = build_checkpointer( # type: ignore
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

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        self.trainer.checkpointer.load(args.train.checkpoint.load_path, state)

        extra = state["extra_state"]
        self.trainer.state.global_step = extra["global_step"]
        self.trainer.start_epoch       = extra["start_epoch"]   # 直接用，不做计算
        self.trainer.start_step        = extra["start_step"]    # 直接用，不做计算

        self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        self.trainer.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        self.trainer.environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "VeOmniArguments" = self.trainer.args

        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")

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
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        helper.empty_cache()
        self.trainer.checkpointer.save(save_checkpoint_path, ckpt_state, save_async=args.train.checkpoint.save_async)

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

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
            self._save_checkpoint(state)

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            self._save_checkpoint(state)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        # self._save_model_assets()
        super().on_train_begin(state)

    def _save_model_assets(self):
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0:
            save_model_assets(args.train.checkpoint.model_assets_dir, self.trainer.model_assets)
        dist.barrier()

    def _save_checkpoint(self, state: TrainerState):
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
            train_architecture=args.train.train_architecture,
            output_dir=args.train.checkpoint.output_dir,
            save_checkpoint_path=save_checkpoint_path,
            model=self.trainer.model,
            fqn_to_index_mapping=args.model.fqn_to_index_mapping,
            is_rank_0=args.train.global_rank == 0,
        )

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()
        logger.info_rank0(f"HF checkpoint saved at {hf_weights_path}")
