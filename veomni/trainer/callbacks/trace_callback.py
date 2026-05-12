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

import time
import os
from typing import TYPE_CHECKING, Any, Dict, List
from collections import deque
from tqdm import trange
import tracemalloc
import contextlib
import torch

from ...distributed.parallel_state import get_parallel_state
from ...utils import helper
from ...utils.dist_utils import all_reduce
from ...utils.logging import get_logger
from .base import Callback, TrainerState


logger = get_logger(__name__)


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


class MoERouterMonitorCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self.monitor = None

        args: "VeOmniArguments" = self.trainer.args
        if not args.train.wandb.enable:
            logger.info_rank0("MoE router monitor disabled (wandb not enabled).")
            return
        if args.train.moe_load_balance_monitor_interval <= 0:
            logger.info_rank0("MoE router monitor disabled (moe_load_balance_monitor_interval=0).")
            return

        config = self.trainer.model_config.foundation_config if getattr(self.trainer.model_config, "foundation_config") else self.trainer.model_config
        if hasattr(config, "num_experts"):
            from ...utils.moe_monitor import MoERouterMonitor, set_active_monitor

            self.monitor = MoERouterMonitor(config.num_experts)
            set_active_monitor(self.monitor)
            logger.info_rank0(
                f"MoE router monitor enabled: num_experts={config.num_experts}, "
                f"interval={args.train.moe_load_balance_monitor_interval}"
            )
        else:
            logger.warning_rank0(
                "moe_load_balance_monitor_interval > 0 but model config has no 'num_experts'. "
                "MoE router monitor not activated."
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if (
            self.monitor
            and state.global_step % args.train.moe_load_balance_monitor_interval == 0
            and args.train.global_rank == 0
        ):
            import wandb

            load_matrix = self.monitor.get_load_matrix(current_step=state.global_step)
            num_layers = load_matrix.shape[0]
            if num_layers == 0:
                logger.warning_rank0(
                    f"Step {state.global_step}: MoE router monitor has no recorded data. "
                    "Check that router forward hooks are registered (e.g. PatchQwen3MoeTopKRouter)."
                )
                return
            from ...utils.moe_monitor import MoERouterMonitor

            image = self.monitor.create_wandb_image(load_matrix)
            vio = MoERouterMonitor.compute_vio(load_matrix)
            max_vio, min_vio, avg_vio = vio["max_vio"], vio["min_vio"], vio["avg_vio"]
            tb_writer = getattr(self.trainer, "tb_writer", None)
            if tb_writer is not None:
            
                for i in range(num_layers):
                    tb_writer.add_scalar(f"moe/max_vio/layer_{i}", max_vio[i].item(), state.global_step)
                    tb_writer.add_scalar(f"moe/min_vio/layer_{i}", min_vio[i].item(), state.global_step)
                    tb_writer.add_scalar(f"moe/avg_vio/layer_{i}", avg_vio[i].item(), state.global_step)
          
                tb_writer.add_scalar("moe/max_vio/max", max_vio.max().item(), state.global_step)
                tb_writer.add_scalar("moe/max_vio/avg", max_vio.mean().item(), state.global_step)
                tb_writer.add_scalar("moe/avg_vio/max", avg_vio.max().item(), state.global_step)
                tb_writer.add_scalar("moe/avg_vio/avg", avg_vio.mean().item(), state.global_step)
        
                heatmap = load_matrix.unsqueeze(0)  # (1, num_layers, num_experts)
            
                tb_writer.add_image("moe/expert_load_heatmap", heatmap, state.global_step)

            metrics = {"moe/expert_load_heatmap": image}
            for i in range(num_layers):
                metrics[f"moe/max_vio/layer_{i}"] = max_vio[i].item()
                metrics[f"moe/min_vio/layer_{i}"] = min_vio[i].item()
                metrics[f"moe/avg_vio/layer_{i}"] = avg_vio[i].item()
            metrics["moe/max_vio/max"] = max_vio.max().item()
            metrics["moe/max_vio/avg"] = max_vio.mean().item()
            metrics["moe/min_vio/max"] = min_vio.max().item()
            metrics["moe/min_vio/avg"] = min_vio.mean().item()
            metrics["moe/avg_vio/max"] = avg_vio.max().item()
            metrics["moe/avg_vio/avg"] = avg_vio.mean().item()
            wandb.log(metrics, step=state.global_step)

            logger.info_rank0(
                f"Step {state.global_step}: uploaded MoE load balance heatmap "
                f"({num_layers} layers, {load_matrix.shape[1]} experts, "
                f"steps {self.monitor._last_step_range[0]}-{self.monitor._last_step_range[1]}), "
                f"max_vio: max={vio['max_vio'].max().item():.4f} avg={vio['max_vio'].mean().item():.4f}, "
                f"min_vio: max={vio['min_vio'].max().item():.4f} avg={vio['min_vio'].mean().item():.4f}, "
                f"avg_vio: max={vio['avg_vio'].max().item():.4f} avg={vio['avg_vio'].mean().item():.4f}."
            )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.monitor is not None:
            from ...utils.moe_monitor import set_active_monitor

            set_active_monitor(None)
            self.monitor = None
            logger.info_rank0("MoE router monitor disabled.")


class WandbTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0 and args.train.wandb.enable:
            from dataclasses import asdict

            import wandb
            try:
                wandb.login(key=os.getenv("WANDB_API_KEY", None))
            except:
                print("未找到 WANDB_API_KEY 环境变量")

            wandb.init(
                project=args.train.wandb.project,
                name=args.train.wandb.name,
                id=args.train.wandb.id,
                resume="allow" if args.train.wandb.id else None,
                config={**asdict(args.model), **asdict(args.data), **asdict(args.train)},
            )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and args.train.wandb.enable:
            import wandb

            wandb.log(self.trainer.step_env_metrics, step=state.global_step)

    def on_eval_end(self, state: TrainerState, eval_metrics: Dict[str, Any] = None, **kwargs) -> None:
        """Log evaluation results to wandb.
 
        Called directly by EvaluateCallback._log_eval_results() immediately after
        evaluation completes, so metrics are always logged at the exact eval step.
 
        Args:
            state:        current TrainerState.
            eval_metrics: dict of metric_name → value, including 'step' and 'epoch'.
        """
        if eval_metrics is None:
            return
 
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank != 0 or not args.train.wandb.enable:
            return
 
        import wandb
 
        # Prefix with "eval/" so training and eval metrics are separated in wandb.
        eval_step = eval_metrics.get("step", state.global_step)
        wandb_payload = {
            f"eval/{k}": v
            for k, v in eval_metrics.items()
            if k not in ("step", "epoch")
        }
        # Also log epoch for easy filtering.
        wandb_payload["eval/epoch"] = eval_metrics.get("epoch", state.epoch)
 
        wandb.log(wandb_payload, step=eval_step)
        logger.info_rank0(
            f"[WandbTrace] Logged {len(wandb_payload)} eval metrics at step={eval_step}."
        )


class TensorboardTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.tb_writer = None
        if args.train.global_rank == 0:
            from torch.utils.tensorboard import SummaryWriter

            tb_log_dir = os.path.join(args.train.checkpoint.output_dir, "runs")
            os.makedirs(tb_log_dir, exist_ok=True)
            
            self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
            self.trainer.tb_writer = self.tb_writer
            print(f"TensorBoard 已经初始化，日志将保存至: {tb_log_dir}")

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args

        if args.train.global_rank == 0 and self.tb_writer is not None:
            for k, v in self.trainer.step_env_metrics.items():
           
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, global_step=state.global_step)

    def on_eval_end(self, state: TrainerState, eval_metrics: Dict[str, Any] = None, **kwargs) -> None:
        """Log evaluation results to TensorBoard."""
        if eval_metrics is None:
            return

        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank != 0 or getattr(self, "tb_writer", None) is None:
            return

        eval_step = eval_metrics.get("step", state.global_step)
        
        # 记录所有非 step/epoch 的 eval 指标
        logged_count = 0
        for k, v in eval_metrics.items():
            if k not in ("step", "epoch") and isinstance(v, (int, float)):
                self.tb_writer.add_scalar(f"eval/{k}", v, global_step=eval_step)
                logged_count += 1
        
        self.tb_writer.flush()
        
        # 如果你的环境中已经配置了 logger，可以解除这行注释
        logger.info_rank0(
            f"[TensorboardTrace] Logged {logged_count} eval metrics at step={eval_step}."
        )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        """训练结束时关闭 TensorBoard Writer"""
        if getattr(self, "tb_writer", None) is not None:
            self.tb_writer.close()
            self.tb_writer = None



class ProfileTraceCallback(Callback):
    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            self.profiler = helper.create_profiler(
                start_step=args.train.profile.start_step,
                end_step=args.train.profile.end_step,
                trace_dir=args.train.profile.trace_dir,
                record_shapes=args.train.profile.record_shapes,
                profile_memory=args.train.profile.profile_memory,
                with_stack=args.train.profile.with_stack,
                global_rank=args.train.global_rank,
            )
            self.profiler.start()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        if args.train.profile.this_rank:
            if state.global_step <= args.train.profile.end_step:
                self.profiler.step()

            if state.global_step == args.train.profile.end_step:
                self.profiler.stop()


class EnvironMeterCallback(Callback):
    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)

        args: "VeOmniArguments" = self.trainer.args
        self.trainer.environ_meter = helper.EnvironMeter(
            config=trainer.model_config,
            global_batch_size=args.train.global_batch_size,
            empty_cache_steps=args.train.empty_cache_steps,
            enable_multisource=args.data.enable_multisource,
            dataloader=trainer.train_dataloader,
            data_path=args.data.train_path,
            gc_steps=args.train.gc_steps,
        )
       
        self._loss_window = deque(maxlen=100)
        self._tracemalloc_started = False
        self._tracemalloc_snapshot = None

    def on_step_begin(self, state: TrainerState, micro_batches: List[List[Dict[str, Any]]] = None, **kwargs) -> None:
        for micro_batch in micro_batches:
            self.trainer.environ_meter.add(micro_batch)
        self.start_time = time.time()

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        delta_time = time.time() - self.start_time
        wq = getattr(self.trainer.train_dataloader, "worker_metrics_queue", None)
        step_env_metrics = self.trainer.environ_meter.step(delta_time, global_step=state.global_step,  worker_metrics_queue=wq)

        step_train_metrics = {
            "loss_avg": loss,
           
        }
        step_train_metrics.update(loss_dict)
        step_train_metrics["grad_norm"] = grad_norm

        # gather training_step_info from all ranks
        step_train_metrics = {
            f"training/{k}": all_reduce(v, group=get_parallel_state().fsdp_group)
            for k, v in step_train_metrics.items()
        }
        step_train_metrics["time_profiling/iter_time"] = delta_time
        current_loss = step_train_metrics["training/loss_avg"]
        self._loss_window.append(current_loss)
        train_loss = sum(self._loss_window) / len(self._loss_window)
        step_train_metrics["training/loss_avg"] = train_loss

        # step_train_metrics["training/raw_loss"] = current_loss
     
        lr = max(self.trainer.lr_scheduler.get_last_lr())
        step_train_metrics["training/lr"] = lr

        step_env_metrics.update(step_train_metrics)

        self.trainer.step_train_metrics = step_train_metrics
        self.trainer.step_env_metrics = step_env_metrics

        if self.trainer.args.train.global_rank != 0:
            return
        
        logging_steps = getattr(self.trainer.args.train, "logging_steps", 10)

        if state.global_step % logging_steps == 0:
            global_loss =  step_train_metrics["training/loss_avg"]
            train_info = (
                f"[step {state.global_step}] "
                f"loss: {global_loss:.4f}  "
                f"grad_norm: {grad_norm:.3f}  "
                f"lr: {lr:.6e}  "
                f"time: {delta_time:.2f}s"
            )
            logger.info(train_info)

            # 吞吐 & 显存
            env_info = (
                f"  mfu: {step_env_metrics.get('system_metric/mfu', 0):.4f}  "
                f"flops_achieved(T): {step_env_metrics.get('system_metric/flops_achieved(T)', 0):.4f}  "
                f"tokens/s: {step_env_metrics.get('training/tokens_per_second(M)', 0):.4f}M  "
                f"consumed: {step_env_metrics.get('training/consume_tokens(B)', 0):.4f}B  "
                f"avg_seqlen: {step_env_metrics.get('training/avg_sample_seq_len', 0):.1f}  "
                f"mem_alloc: {step_env_metrics.get('memory/max_memory_allocated(GB)', 0):.2f}GB  "
                f"mem_reserved: {step_env_metrics.get('memory/max_memory_reserved(GB)', 0):.2f}GB  "
                f"alloc_retries: {step_env_metrics.get('memory/num_alloc_retries', 0)}  "
                f"cpu_memory_usage(%): {step_env_metrics.get('memory/cpu_memory_usage(%)', 0):.2f} "
            )
            logger.info(env_info)
        
        if state.global_step == 50:
            tracemalloc.start(10)  # 10层调用栈
            self._tracemalloc_snapshot = tracemalloc.take_snapshot()
            logger.info("[MemTrace] baseline snapshot taken.")
        
        elif state.global_step % 100 == 0 and self._tracemalloc_snapshot is not None:
            # 之后每 100 步和基准对比
            new_snapshot = tracemalloc.take_snapshot()
            top_stats = new_snapshot.compare_to(
                self._tracemalloc_snapshot, 'lineno'
            )
            logger.info(f"[MemTrace] step={state.global_step} top memory growth:")
            for stat in top_stats[:10]:  
                logger.info(f"  {stat}")
           
            self._tracemalloc_snapshot = new_snapshot


class TqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.data_loader_tqdm = trange(
            args.train_steps,
            desc=f"Epoch {state.epoch + 1}/{args.train.num_train_epochs}",
            total=args.train_steps,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )
        

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        postfix = ", ".join(f"{k.split('/', 1)[-1]}: {v:.2f}" for k, v in self.trainer.step_train_metrics.items())
        self.data_loader_tqdm.set_postfix_str(postfix)
        self.data_loader_tqdm.update()


class VideoTqdmCallback(Callback):
    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        args: "VeOmniArguments" = self.trainer.args
        self.epoch_total = self.trainer.init_data_size
        self._last_video_trained_num = self.trainer.start_step  # 恢复训练时的偏移
        self.data_loader_tqdm = trange(
            self.epoch_total,
            desc=f"data size {self.epoch_total}",
            total=self.epoch_total,
            initial=self.trainer.start_step,
            disable=args.train.local_rank != 0,
        )

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        self.data_loader_tqdm.close()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        def fmt(k, v):
            if "loss" in k:
                return f"{k.split('/', 1)[-1]}: {v:.4f}"   # loss 保留4位
            elif "grad_norm" in k:
                return f"{k.split('/', 1)[-1]}: {v:.3f}"   # grad_norm 保留3位
            elif "lr" in k:
                return f"{k.split('/', 1)[-1]}: {v:.3e}"   # lr 用科学计数法
            elif "iter_time" in k:
                return f"{k.split('/', 1)[-1]}: {v:.2f}"
            else:
                return f"{k.split('/', 1)[-1]}: {v:.4f}"
            
        trained_videos = state.video_trained_num
        global_step = f"step:{state.global_step}"
        epoch_progress = f"{trained_videos / self.epoch_total:.3f}" if self.epoch_total > 0 else "0.000"

        metrics_str = ", ".join(fmt(k, v) for k, v in self.trainer.step_train_metrics.items())
        eval_metrics: dict = getattr(self.trainer, "eval_metrics", None)
        eval_suffix = ""
        if eval_metrics:
            eval_step = eval_metrics.get("step", "?")
            core = {k: v for k, v in eval_metrics.items() if k not in ("step", "epoch")}
            # Show at most the 4 most important metrics to keep the bar readable.
            priority_keys = ["acc", "prob", "all_acc"] + [
                k for k in core if k not in ("acc", "prob", "all_acc")
            ]
            shown = {k: core[k] for k in priority_keys[:4] if k in core}
            eval_summary = " ".join(f"{k}:{v:.3f}" for k, v in shown.items())
            eval_suffix = f" | eval@{eval_step}[{eval_summary}]"
 
        postfix = f"{epoch_progress} | {global_step} | {metrics_str} | {eval_suffix}"
        self.data_loader_tqdm.set_postfix_str(postfix, refresh=False)
        delta = state.video_trained_num - self._last_video_trained_num
        self.data_loader_tqdm.update(max(delta, 1))
        self._last_video_trained_num = state.video_trained_num
        self.trainer.eval_metrics = None

class StepTimer:
    def __init__(self):
        self.events = {}

    @contextlib.contextmanager
    def measure(self, name):
      
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
      
        end.record()
        if name not in self.events:
            self.events[name] = []
        self.events[name].append((start, end))

    def get_and_reset(self) -> Dict[str, float]:
       
        torch.cuda.synchronize()
        timings = {}
        for name, evts in self.events.items():
            total_ms = sum(s.elapsed_time(e) for s, e in evts)
            timings[name] = total_ms / 1000.0  # 转换为秒
        self.events.clear()
        return timings

class ComponentTimingCallback(Callback):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.step_timer = StepTimer()
        self.logging_steps = getattr(trainer.args.train, "logging_steps", 10)

    def on_step_begin(self, state: TrainerState, **kwargs) -> None:
       
        if state.global_step % self.logging_steps == 0:
            self.trainer._current_step_timer = self.step_timer
        else:
            self.trainer._current_step_timer = None

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if getattr(self.trainer, "_current_step_timer", None) is not None:
            timings = self.trainer._current_step_timer.get_and_reset()
            for k, v in timings.items():
                metric_name = f"time_profiling/time_{k}"
                self.trainer.step_env_metrics[metric_name] = v
                self.trainer.step_train_metrics[metric_name] = v