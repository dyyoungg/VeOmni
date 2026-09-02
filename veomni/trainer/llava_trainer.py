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

from inspect import Traceback
import os
import json
from abc import ABC
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional
import contextlib

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import set_checkpoint_debug_enabled
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor
)
from transformers.modeling_outputs import ModelOutput

from veomni.arguments import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    VeOmniArguments,
    save_args,
)
from veomni.data.llavaomni_dataloader import get_eval_dataloader, get_train_dataloader
from veomni.data.ulysess_dataloader import make_ulysses_train_dataloader
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, build_tokenizer
from veomni.models.custom.llava_qwen3moe.auto import build_qwen3moe_omni_from_pretrained
from veomni.models.custom.llava_qwen2.auto import build_llavaqwen2_omni_from_pretrained
from veomni.models.custom.llava_qwen35.auto import build_qwen35_omni_from_pretrained
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.trainer.callbacks import (
    CheckpointerCallback,
    EnvironMeterCallback,
    EvaluateCallback,
    HuggingfaceCkptCallback,
    MoERouterMonitorCallback,
    ProfileTraceCallback,
    VideoTqdmCallback,
    TrainerState,
    WandbTraceCallback,
    TensorboardTraceCallback,
    ComponentTimingCallback
)
from veomni.utils import helper, logging
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.loss_utils import count_loss_token, mean_global_loss
from veomni.utils.model_utils import pretty_print_trainable_parameters
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel.comm import get_data_parallel_world_size
from veomni.ops.batch_invariant_ops import set_batch_invariant_mode
from veomni.utils.constants import get_image_video_audio_placeholder
from veomni.utils.constants import IGNORE_INDEX


logger = helper.create_logger(__name__)

@dataclass
class VLMTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    freeze_vit_projector: bool = field(default=False, metadata={"help": "Whether or not to freeze the vit projector parameters."})
    freeze_audio_tower: bool = field(default=False, metadata={"help": "Whether or not to freeze the audio tower parameters."})
    freeze_audio_projector: bool = field(default=False, metadata={"help": "Whether or not to freeze the audio tower parameters."})
    freeze_llm: bool = field(default=False, metadata={"help": "Whether or not to freeze the llm parameters."})
    freeze_router: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the MoE router (gate) parameters, keeping the pretrained routing fixed while experts/attention still train."},
    )
    vit_chunk_forward: bool = field(
        default=False,
        metadata={"help": "Enable chunked ViT forward to reduce peak activation memory. "
                  "Images are processed in chunks of vit_chunk_size through the ViT backbone, "
                  "then all features are passed through the projector at once."},
    )
    vit_chunk_size: int = field(
        default=20,
        metadata={"help": "Max number of images per ViT forward chunk. "
                  "Only effective when vit_chunk_forward=True. "
                  "Under FSDP, ranks auto-align forward count via one scalar all-reduce."},
    )

    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )
    pack_seq: bool = field(
        default=True,
        metadata={"help": "Whether or not to pack seq to max len."},
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataloader_num_workers: int = field(default=2, metadata={"help": "dataloder workers"})
    dataloader_prefetch_factor: int = field(
        default=2,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    dataloader_debug: bool = field(default=False)
    fix_image_size: bool = field(default=False)
    jpeg_image_augmentation: bool = field(default=False)
    audio_volume_augmentation: bool = field(
        default=True,
        metadata={"help": "Randomly adjust audio volume during training to improve robustness to quiet/loud audio."},
    )
    audio_volume_gain_min_db: float = field(
        default=-10.0,
        metadata={"help": "Min volume gain in dB for audio augmentation (negative = attenuate)."},
    )
    audio_volume_gain_max_db: float = field(
        default=10.0,
        metadata={"help": "Max volume gain in dB for audio augmentation (positive = amplify)."},
    )
    audio_volume_augmentation_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of applying audio volume augmentation per sample."},
    )
    video_decode_method: str = field(default="decord")
    remote_dataloader: bool = field(default=False)
    use_fake_data: bool = field(
        default=False,
        metadata={"help": "Use randomly generated fake multimodal data for training speed benchmarking. "
                  "No real data files are needed when this is enabled."},
    )
    fake_data_num_samples: int = field(
        default=10000,
        metadata={"help": "Number of fake samples to generate when use_fake_data=True."},
    )
    proactive: bool = field(default=False)
    target_image_num: int = field(default=999)
    min_lr_rate: float = field(default=0.0)
    router_aux_loss_coef: float = field(default=0.001)
    output_router_logits: bool = field(default=True)
    mm_balance_coef: float = field(
        default=0.0,
        metadata={"help": "Coefficient for multimodal token load-balancing aux loss relative to text. "
                  "1.0 = same as text (original behaviour), 0.0 = text-only balancing, 0.1 = light constraint to prevent expert collapse."},
    )
    modality_aware_routing: bool = field(
        default=False,
        metadata={"help": "When True, add a learnable per-modality bias to MoE router logits so the gate can distinguish text/vision/audio tokens."},
    )
    num_routing_modalities: int = field(
        default=3,
        metadata={"help": "Number of modality types for modality-aware routing (default 3: text=0, vision=1, audio=2)."},
    )
    logging_steps: int = field(default=10, metadata={"help": "Log every N steps"})
    eval_first: bool = field(default=True)
    save_predictions: bool = field(default=True)
    channel_loss_enable: bool = field(default=False, metadata={"help": "Enable per-channel loss logging."})
    channel_loss_interval: int = field(default=2, metadata={"help": "Compute channel loss every N steps."})
    channel_loss_mapping: str = field(default="", metadata={"help": "Path to JSON file mapping dataset_type -> channel name."})

@dataclass
class VLMMDataArguments(DataArguments):
    mm_configs: Optional[Dict] = field(
        default_factory=dict,
        metadata={"help": "Config for multimodal input."},
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    offline_dataset_split: bool = field(default=False)
    finetune_sample_frames: int = field(default=-1)
    save_token_counted_data: bool = field(default=False)
    tokencounted_data_save_dir: Optional[str] = field(default="/mnt/afs/data")
    preprocess_workers: int = field(default=2)
    sample_fps: int = field(default=4)
    use_finetune_fps: bool= field(default=False)
    offset_file_path: str = field(default="")
    file_maping_path: str = field(default="")
    
    # subtitle_video_config
    max_subtitle_token_num: int = field(default=2048)
    high_resolution_interval: float = field(default=0.0)
    sub_native_resolution: bool = field(default=False)
    compress_fps: bool = field(default=False)
    minimum_image_token_num: int = field(default=10)
    keep_last_n_image_turns: int = field(default=0)


@dataclass
class VLMMModelArguments(ModelArguments):
    encoder_data_balance: Optional[bool] = field(
        default=False, metadata={"help": "Whether to balance encoder data for qwen3-vl model"}
    )
    encoder_data_balance_sorting_algo: Optional[str] = field(
        default="post_mbs_balancing_greedy_without_pad",
        metadata={
            "help": "The sorting algorithm of encoder data balance. All viable algorithms are defined in "
            "veomni/utils/data_balance/balance_sorting_algo.py, SORTING_ALGO_FUNC"
        },
    )
    model_arc: Optional[str] = field(default="qwen2")

    vision_tower: Optional[List[str]] = field(default=None)
    mm_downsample_ratio: int = field(default=16)
    dynamic_downsample: bool = field(default=False, metadata={"help": "Enable dynamic downsample ratio during training"})
    dynamic_downsample_ratios: List[int] = field(default_factory=lambda: [4, 8, 12], metadata={"help": "Candidate downsample ratios for dynamic selection"})
    audio_downsample_ratio: int = field(default=10)
    audio_frame_length: int = field(default=320)
    num_mel_bins: Optional[int] = field(default=128)
    audio_max_duration: Optional[int] = field(default=30)
    use_audio_start_end_token: bool = field(default=False)
    mm_image_size: List[int] = field(default_factory=lambda: [644, 364])  # 固定的图片尺寸: [width, height]
    image_projector_type: Optional[str] = field(default="avgpool")
    audio_projector_type: Optional[str] = field(default="conv_channel_upscale") # avgpool, channel_upscale
    audio_encoder_type: Optional[str] = field(default="whisper")
    qwen3_audio_n_window: Optional[int] = field(default=50)
    use_3d_rope: bool = field(
            default=False,
            metadata={"help": "Enable 3D M-RoPE position IDs pre-computation in the dataloader (for Qwen3.5-based models)."},
    )

@dataclass
class VeOmniVLMArguments(VeOmniArguments):
    model: "VLMMModelArguments" = field(default_factory=VLMMModelArguments)
    data: "VLMMDataArguments" = field(default_factory=VLMMDataArguments)
    train: "VLMTrainingArguments" = field(default_factory=VLMTrainingArguments)


class VLMTrainer:

    def __init__(self, args: VeOmniVLMArguments):
  
        self.args = args

        self._setup()
        # rewrite build model to support data balancing
        self._build_model()
        self._build_model_assets()
        # rewrite freeze_model_module to support freeze multimodal encoder, etc.
        self._build_dataloader()
        self._freeze_model_module()
        self._build_parallelized_model()
       

        # rewrite build_optimizer to support different lr param groups
        self._build_optimizer()
        self._init_callbacks()
        self._build_lr_scheduler()
        self._build_training_context()
   

    def _setup(self):
        # log args
        # logger.info_rank0(json.dumps(asdict(self.args), indent=2))
        # init distributed environment
        device_str = f"{get_device_type()}:{self.args.train.local_rank}"
 
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        # Initialize distributed process group
        if not dist.is_initialized():
            dist.init_process_group(backend=get_dist_comm_backend(),
                                    device_id=torch.device(device_str))
       
        logger.info(f"Process rank: {self.args.train.global_rank}, {device_str}, world size: {self.args.train.world_size}")

        # Initialize parallel state
        init_parallel_state(
            dp_size=self.args.train.accelerator.dp_size,
            dp_replicate_size=self.args.train.accelerator.dp_replicate_size,
            dp_shard_size=self.args.train.accelerator.dp_shard_size,
            tp_size=self.args.train.accelerator.tp_size,
            pp_size=self.args.train.accelerator.pp_size,
            cp_size=self.args.train.accelerator.cp_size,
            ulysses_size=self.args.train.accelerator.ulysses_size,
            extra_parallel_sizes=self.args.train.accelerator.extra_parallel_sizes,
            extra_parallel_placement_innermost=self.args.train.accelerator.extra_parallel_placement_innermost,
            extra_parallel_names=self.args.train.accelerator.extra_parallel_names,
            dp_mode=self.args.train.accelerator.fsdp_config.fsdp_mode,
            async_enabled=self.args.train.accelerator.enable_async,
        )

        # Set random seed
        helper.set_seed(self.args.train.seed, self.args.train.enable_full_determinism)

        # Enable high precision for bf16
        helper.enable_high_precision_for_bf16()

        # Enable third party logging
        if self.args.train.local_rank == 0:
            helper.enable_third_party_logging()

        # Save arguments
        if self.args.train.global_rank == 0:
            save_args(self.args, self.args.train.checkpoint.output_dir)

        # Gradient checkpointing debug
        set_checkpoint_debug_enabled(self.args.train.gradient_checkpointing.debug)

    def _build_model(self):
        args: VeOmniVLMArguments = self.args
        logger.info_rank0("Build model")
        self.model_config = AutoConfig.from_pretrained(args.model.config_path, trust_remote_code=True)

        if self.model_config.model_type == "llavaqwen3moe_omni":
            self.model = build_qwen3moe_omni_from_pretrained(
                args.model.model_path,
                init_device=args.train.init_device,
                torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                encoder_data_balance=args.model.encoder_data_balance,
                encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
                modality_aware_routing=args.train.modality_aware_routing,
                num_routing_modalities=args.train.num_routing_modalities,
                ops_implementation=args.model.ops_implementation,
            )
        elif self.model_config.model_type == "llavaqwen2_omni":
            self.model = build_llavaqwen2_omni_from_pretrained(
                args.model.model_path,
                init_device=args.train.init_device,
                torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                encoder_data_balance=args.model.encoder_data_balance,
                encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
                ops_implementation=args.model.ops_implementation,
            )
        elif self.model_config.model_type == "llavaqwen35_omni":
            self.model = build_qwen35_omni_from_pretrained(
                args.model.model_path,
                init_device=args.train.init_device,
                torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                encoder_data_balance=args.model.encoder_data_balance,
                encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
                ops_implementation=args.model.ops_implementation,
            )

        else:
            self.model = build_foundation_model(
                config_path=args.model.config_path,
                weights_path=args.model.model_path,
                torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
                init_device=args.train.init_device,
                encoder_data_balance=args.model.encoder_data_balance,
                encoder_data_balance_sorting_algo=args.model.encoder_data_balance_sorting_algo,
                ops_implementation=args.model.ops_implementation,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model.config_path,
                                                       model_max_length=self.args.train.model_max_length,
                                                       padding_side="right",
                                                       use_fast=True,)

        image_token_id, video_token_id, audio_token_id = get_image_video_audio_placeholder(self.tokenizer)

        self.model.omni_config.image_token_id = image_token_id
        self.model.omni_config.video_token_id = video_token_id
        self.model.omni_config.audio_token_id = audio_token_id
        logger.info_rank0(f"image pad token {image_token_id}, video pad token: {video_token_id}, audio pad token:{audio_token_id}")
        if self.model_config.model_type in ("llavaqwen3moe_omni", "llavaqwen35_omni"):
            self.model.config.output_router_logits = self.args.train.output_router_logits
            self.model.foundation_config.router_aux_loss_coef = self.args.train.router_aux_loss_coef
            self.model.omni_config.mm_balance_coef = self.args.train.mm_balance_coef
            self.model.omni_config.modality_aware_routing = self.args.train.modality_aware_routing
        self.model_config.freeze_vit = self.args.train.freeze_vit
        self.model_config.freeze_audio = self.args.train.freeze_audio_tower
        self.model_config.freeze_audio_projector = self.args.train.freeze_audio_projector
        self.model_config.freeze_llm = self.args.train.freeze_llm
        self.model_config.gradient_checkpointing = self.args.train.gradient_checkpointing.enable
        self.model_config.gradient_checkpointing_vit = (
            self.args.train.gradient_checkpointing.enable and self.args.train.gradient_checkpointing.enable_vit
        )
        self.model.config.encoder_data_balance = self.args.model.encoder_data_balance
        self.model.config.encoder_data_balance_sorting_algo = self.args.model.encoder_data_balance_sorting_algo

        # Chunked ViT forward config
        if self.args.train.vit_chunk_forward:
            self.model.image_encoder.vit_chunk_size = self.args.train.vit_chunk_size
            logger.info_rank0(f"Enabled chunked ViT forward with chunk_size={self.args.train.vit_chunk_size}")
        else:
            self.model.image_encoder.vit_chunk_size = 0

    def _freeze_model_module(self):
        args: VeOmniVLMArguments = self.args
        model_config = self.model_config

        if model_config.model_type in ("llavaqwen3moe_omni", "llavaqwen2_omni", "llavaqwen35_omni"):
            if args.train.freeze_vit:
                self.model.image_encoder.requires_grad_(False)
                self.model.image_encoder.freeze_vit = True
                self.model.image_encoder.mm_projector.requires_grad_(True)
           
            if args.train.freeze_audio_tower:
                self.model.audio_encoder.requires_grad_(False)
                self.model.audio_encoder.freeze_audio_encoder = True
                self.model.audio_encoder.audio_projector.requires_grad_(True)
            
            if args.train.freeze_vit_projector:
                self.model.image_encoder.mm_projector.requires_grad_(False)
                self.model.image_encoder.freeze_image_projector = True

            if args.train.freeze_audio_projector:
                self.model.audio_encoder.audio_projector.requires_grad_(False)
                self.model.audio_encoder.freeze_audio_projector = True

            if args.train.freeze_llm:
                self.model.model.requires_grad_(False)
                self.model.lm_head.requires_grad_(False)

            if args.train.freeze_router:
                # Freeze every MoE router (gate) so the pretrained routing stays fixed
                # while experts/attention continue to train. Router params live at
                # `model.layers.<i>.mlp.gate.weight` (Qwen3MoeSparseMoeBlock.gate).
                num_frozen = 0
                for name, param in self.model.named_parameters():
                    if name.endswith("mlp.gate.weight"):
                        param.requires_grad_(False)
                        num_frozen += 1
                    if name.endswith("mlp.gate.modality_bias") and self.model.omni_config.modality_aware_routing:
                        
                        param.requires_grad_(False)
                        num_frozen += 1
                logger.info_rank0(f"freeze_router enabled: froze {num_frozen} MoE router (gate) params.")
            else:
                num_frozen = 0
                for name, param in self.model.named_parameters():
                    if name.endswith("mlp.gate.weight"):
                        param.requires_grad_(True)
                        num_frozen += 1
                    if name.endswith("mlp.gate.modality_bias") and self.model.omni_config.modality_aware_routing:
                        param.requires_grad_(True)
                logger.info_rank0(f"unfreeze router enabled: unfreeze {num_frozen} MoE router (gate) params.")
        else:
            raise NotImplementedError(f"{model_config.model_type} is not supported now.")
           
        pretty_print_trainable_parameters(self.model)
        helper.print_device_mem_info("VRAM usage after building model")


    def _build_dataloader(self, ):

        args: VeOmniArguments = self.args

        tokenizer = self.tokenizer
        training_args, model_args, data_args = args.train, args.model, args.data
        if get_parallel_state() is not None and get_parallel_state().ulysses_enabled:
            self.train_dataloader = make_ulysses_train_dataloader(data_args, training_args, model_args, tokenizer)
        else:
            self.train_dataloader = get_train_dataloader(data_args, training_args, model_args, tokenizer)
        dist.barrier()
        
        if getattr(training_args, "use_fake_data", False):
            self.eval_dataloader = None
        else:
            self.eval_dataloader = get_eval_dataloader(tokenizer, data_args, training_args, model_args)
    
    def _build_model_assets(self):
        args: VeOmniVLMArguments = self.args
        self.processor = AutoProcessor.from_pretrained(args.model.vision_tower)
        self.model_assets = [self.model_config, self.processor, self.tokenizer]
        # generation_config.json
        try:
            from transformers import GenerationConfig
            generation_config = GenerationConfig.from_pretrained(args.model.config_path)
            self.model_assets.append(generation_config)
        except Exception:
            pass
        
    def _build_parallelized_model(self):
        args: VeOmniArguments = self.args
        # Parallelize model
        self.model = build_parallelize_model(
            self.model,
            init_device=args.train.init_device,
            weights_path=args.model.model_path,
            enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
            mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
            enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
            enable_gradient_checkpointing_vit=args.train.gradient_checkpointing.enable_vit,
            enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
            basic_modules=list(
                set(getattr(self.model, "_no_split_modules", None) or []) | set(args.model.basic_modules)
            ),
            enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
            enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
        )
        
        self.model.train()
       
        

    def _build_optimizer(self):
        args: VeOmniVLMArguments = self.args

        vit_params, audio_params, llm_params = [], [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "image_encoder" in name:
                    vit_params.append(param)

                elif "audio_encoder" in name:
                    audio_params.append(param)

                else:
                    llm_params.append(param)

        param_groups = [
            {"params": vit_params, "lr": args.train.vit_lr},
            {"params": audio_params, "lr": args.train.vit_lr},
            {"params": llm_params, "lr": args.train.optimizer.lr},
        ]
        # Drop empty param groups. An empty group (e.g. when the audio tower and
        # projector are both frozen) cannot be mapped back by DCP's
        # set_optimizer_state_dict on resume (it matches groups by param FQN), so
        # its hyperparams (betas/eps/...) get stripped and Adam.step() then raises
        # `KeyError: 'betas'`. Keeping only non-empty groups avoids this.
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        # Build optimizer
        self.optimizer = build_optimizer(
            self.model,
            lr=args.train.optimizer.lr,
            weight_decay=args.train.optimizer.weight_decay,
            fused=True,
            optimizer_type=args.train.optimizer.type,
            param_groups=param_groups,
            no_decay_modules=args.train.optimizer.no_decay_modules,
            no_decay_params=args.train.optimizer.no_decay_params,
            optimizer_config=args.train.optimizer,
        )

    def _build_lr_scheduler(self):
        args: VeOmniArguments = self.args
        # Build lr scheduler
        if get_parallel_state() is not None:
            dp_world_size = get_data_parallel_world_size()
        else:
            dp_world_size = int(os.environ.get('WORLD_SIZE', 1))
        if args.train.remote_dataloader:
            self.init_data_size = len(self.train_dataloader.data_list)
        elif hasattr(self.train_dataloader, "num_train_epochs"):
            # Ulysses PrefetchingPackedLoader: data_list is single-epoch per DP rank.
            # Total samples across all epochs = per_rank * dp_world_size * num_epochs.
            self.init_data_size = (
                len(self.train_dataloader.data_list)
                * dp_world_size
                * self.train_dataloader.num_train_epochs
            )
        else:
            # OmniDataloader: data_list already contains num_epochs copies
            self.init_data_size = len(self.train_dataloader.data_list) * dp_world_size

        # Fake data mode: use max_steps to control training loop length.
        # Workers generate data infinitely; the loop terminates by step count.
        if getattr(args.train, "use_fake_data", False):
            max_steps = getattr(args.train, "max_steps", None)
            if max_steps is not None:
                self.init_data_size = max_steps
            else:
                # Fallback: fake_data_num_samples controls loop length
                self.init_data_size = getattr(args.train, "fake_data_num_samples", 10000)
       
        self.start_step = 0
        self.video_trained_num = 0  
        self.train_steps = self.init_data_size
        print("dp world size", dp_world_size, "Total initial data size", self.init_data_size)
        
        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=self.train_steps,
            lr=args.train.optimizer.lr,
            lr_min=args.train.optimizer.lr_min,
            lr_decay_style=args.train.optimizer.lr_decay_style,
            lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
            lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
            lr_start=args.train.optimizer.lr_start,
        )

    def _build_training_context(self):
        """Build training context for distributed training."""
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.args.train.accelerator.offload_config.enable_activation,
            self.args.train.gradient_checkpointing.enable,
            self.args.train.accelerator.offload_config.activation_gpu_limit,
        )

    # ------------------------------------------------------------------
    # Channel Loss
    # ------------------------------------------------------------------
    def _init_channel_loss(self):
        """Initialize per-channel loss tracking."""
        self._channel_mapping: Dict[str, str] = {}
        self._channel_loss_hs = None
        self._channel_loss_lm_weight = None

        if not self.args.train.channel_loss_enable:
            return

        # Load mapping file
        mapping_path = self.args.train.channel_loss_mapping
        if mapping_path and os.path.isfile(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                self._channel_mapping = json.load(f)
            logger.info(f"[ChannelLoss] Loaded mapping with {len(self._channel_mapping)} entries from {mapping_path}")
        elif mapping_path:
            logger.warning(f"[ChannelLoss] Mapping file not found: {mapping_path}, all samples map to 'other'")

        # Wrap model.loss_function to capture hidden_states and lm_head.weight
        # from within the forward pass, where FSDP2 has already all-gathered params.
        # Same approach as main branch ChannelLossCallback._wrapped_loss_fn.
        inner = self.model
        while hasattr(inner, 'module'):
            inner = inner.module
        if not hasattr(inner, 'loss_function') and hasattr(inner, 'thinker'):
            inner = inner.thinker

        if hasattr(inner, 'loss_function'):
            original_loss_fn = inner.loss_function

            def _wrapped_loss_fn(*args, **kwargs):
                result = original_loss_fn(*args, **kwargs)
                # Only capture on interval steps
                if self.state.global_step % self.args.train.channel_loss_interval == 0:
                    hs = kwargs.get('hidden_states')
                    w = kwargs.get('weights')
                    if hs is None and len(args) > 3:
                        hs = args[3]  # positional fallback
                    if w is None and len(args) > 4:
                        w = args[4]
                    if hs is not None and w is not None:
                        self._channel_loss_hs = hs.detach().clone()
                        self._channel_loss_lm_weight = w.detach().clone()
                return result

            inner.loss_function = _wrapped_loss_fn
            logger.info(f"[ChannelLoss] Wrapped {type(inner).__name__}.loss_function")
        else:
            logger.warning("[ChannelLoss] Cannot find loss_function, channel loss disabled")

        logger.info(f"[ChannelLoss] Enabled, interval={self.args.train.channel_loss_interval}")

        # Pass mapping to dataloader so worker processes can tag samples
        if hasattr(self, 'train_dataloader') and hasattr(self.train_dataloader, 'channel_mapping'):
            self.train_dataloader.channel_mapping = self._channel_mapping

    @torch.no_grad()
    def _compute_channel_loss(
        self,
        hidden_states: torch.Tensor,
        lm_weight: torch.Tensor,
        labels: torch.Tensor,
        channel_ids: List[str],
        seq_lens: List[int],
    ) -> Dict[str, float]:
        """Compute detached per-channel CE loss from hidden_states + lm_head weight.

        Both tensors are captured during lm_head forward (FSDP2 all-gathered),
        so no DTensor / full_tensor() needed here.
        """
        hs = hidden_states.reshape(-1, hidden_states.size(-1))  # [L, D]
        labs = labels.reshape(-1)  # [L]

        shifted_hs = hs[:-1]      # [L-1, D]
        shifted_labs = labs[1:]    # [L-1]

        channel_sums: Dict[str, float] = defaultdict(float)
        channel_counts: Dict[str, int] = defaultdict(int)

        CHUNK = 1024
        offset = 0
        for i, seg_len in enumerate(seq_lens):
            ch = channel_ids[i] if i < len(channel_ids) else "other"
            seg_end = min(offset + seg_len, len(shifted_hs))

            for j in range(offset, seg_end, CHUNK):
                end = min(j + CHUNK, seg_end)
                chunk_logits = F.linear(shifted_hs[j:end].float(), lm_weight.float())
                chunk_loss = F.cross_entropy(
                    chunk_logits,
                    shifted_labs[j:end].to(chunk_logits.device),
                    ignore_index=IGNORE_INDEX,
                    reduction='none',
                )
                valid = shifted_labs[j:end] != IGNORE_INDEX
                channel_sums[ch] += chunk_loss[valid].sum().item()
                channel_counts[ch] += valid.sum().item()
                del chunk_logits, chunk_loss

            offset += seg_len

        result: Dict[str, float] = {}
        for ch in sorted(channel_sums):
            result[f"channel_loss_sum/{ch}"] = channel_sums[ch]
            result[f"channel_token_count/{ch}"] = float(channel_counts[ch])
        return result

    def _init_callbacks(self):
        """Initialize callbacks."""
        self.environ_meter_callback = EnvironMeterCallback(self)
        self.timing_callback = ComponentTimingCallback(self)
        self.tqdm_callback = VideoTqdmCallback(self)
        self.wandb_callback = WandbTraceCallback(self)
        self.tensorboard_callback = TensorboardTraceCallback(self)
        self.profile_callback = ProfileTraceCallback(self)
        self.checkpointer_callback = CheckpointerCallback(self)
        self.hf_ckpt_callback = HuggingfaceCkptCallback(self)
        self.evaluate_callback = EvaluateCallback(self)
        self.moe_monitor_callback = MoERouterMonitorCallback(self)
        self.state = TrainerState()
        self._init_channel_loss()

    def on_train_begin(self):
        self.environ_meter_callback.on_train_begin(self.state)
        self.tqdm_callback.on_train_begin(self.state)
        self.wandb_callback.on_train_begin(self.state)
        self.tensorboard_callback.on_train_begin(self.state)
        self.profile_callback.on_train_begin(self.state)
        self.checkpointer_callback.on_train_begin(self.state)
        self.hf_ckpt_callback.on_train_begin(self.state)
        self.evaluate_callback.on_train_begin(self.state)
        self.moe_monitor_callback.on_train_begin(self.state)

    def on_train_end(self):
        self.environ_meter_callback.on_train_end(self.state)
        self.tqdm_callback.on_train_end(self.state)
        self.wandb_callback.on_train_end(self.state)
        self.tensorboard_callback.on_train_end(self.state)
        self.profile_callback.on_train_end(self.state)
        self.checkpointer_callback.on_train_end(self.state)
        self.hf_ckpt_callback.on_train_end(self.state)
        self.evaluate_callback.on_train_end(self.state)
        self.moe_monitor_callback.on_train_end(self.state)

    def on_epoch_begin(self):
        self.environ_meter_callback.on_epoch_begin(self.state)
        self.tqdm_callback.on_epoch_begin(self.state)
        self.wandb_callback.on_epoch_begin(self.state)
        self.tensorboard_callback.on_epoch_begin(self.state)
        self.profile_callback.on_epoch_begin(self.state)
        self.checkpointer_callback.on_epoch_begin(self.state)
        self.hf_ckpt_callback.on_epoch_begin(self.state)
        self.evaluate_callback.on_epoch_begin(self.state)

    def on_epoch_end(self):

        self.environ_meter_callback.on_epoch_end(self.state)
        self.tqdm_callback.on_epoch_end(self.state)
        self.wandb_callback.on_epoch_end(self.state)
        self.tensorboard_callback.on_epoch_end(self.state)
        self.profile_callback.on_epoch_end(self.state)
        self.checkpointer_callback.on_epoch_end(self.state)
        self.hf_ckpt_callback.on_epoch_end(self.state)
        self.evaluate_callback.on_epoch_end(self.state)

    def on_step_begin(self, micro_batches=None):
        self.environ_meter_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.timing_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.tqdm_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.wandb_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.tensorboard_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.profile_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.checkpointer_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.hf_ckpt_callback.on_step_begin(self.state, micro_batches=micro_batches)
        self.evaluate_callback.on_step_begin(self.state, micro_batches=micro_batches)

    def on_step_end(self, loss=None, loss_dict=None, grad_norm=None):
        self.environ_meter_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.timing_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.tqdm_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.wandb_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.tensorboard_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.profile_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.checkpointer_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.hf_ckpt_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.evaluate_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)
        self.moe_monitor_callback.on_step_end(self.state, loss=loss, loss_dict=loss_dict, grad_norm=grad_norm)

    def preforward(self, micro_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess micro batches before forward pass."""
        micro_batch = {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in micro_batch.items()
        }
        if getattr(self, "LOG_SAMPLE", False):
            helper.print_example(example=micro_batch, rank=self.args.train.local_rank)
            self.LOG_SAMPLE = False
        return micro_batch

    def postforward(
        self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Postprocess model outputs after forward pass."""
        loss_dict: Dict[str, torch.Tensor] = mean_global_loss(
            outputs.loss, self.micro_batch_token_len, self.micro_batches_token_len
        )
        loss = torch.stack(list(loss_dict.values())).sum()
        if getattr(outputs, "aux_loss", None) is not None:
            loss_dict["aux_loss"] = outputs.aux_loss

        # Per-channel loss (detached, no impact on training)
        if (
            self.args.train.channel_loss_enable
            and self.state.global_step % self.args.train.channel_loss_interval == 0
            and self._channel_loss_hs is not None
            and self._channel_loss_lm_weight is not None
        ):
            channel_ids = micro_batch.get("channel_ids", [])
            seq_lens_tensor = micro_batch.get("seq_lens")
            attn_mask_len = micro_batch.get("attention_mask_len")
            if seq_lens_tensor is not None:
                seq_lens = seq_lens_tensor.tolist() if isinstance(seq_lens_tensor, torch.Tensor) else seq_lens_tensor
            elif attn_mask_len is not None:
                seq_lens = attn_mask_len if isinstance(attn_mask_len, list) else attn_mask_len.tolist()
            else:
                seq_lens = [micro_batch["labels"].numel()]

            if channel_ids:
                import time as _time
                _t0 = _time.monotonic()
                channel_metrics = self._compute_channel_loss(
                    hidden_states=self._channel_loss_hs,
                    lm_weight=self._channel_loss_lm_weight,
                    labels=micro_batch["labels"],
                    channel_ids=channel_ids,
                    seq_lens=seq_lens,
                )
                channel_metrics["channel_loss/time"] = _time.monotonic() - _t0
                loss_dict.update(channel_metrics)

            self._channel_loss_hs = None
            self._channel_loss_lm_weight = None

        return loss, loss_dict

    def forward_backward_step(
        self, micro_batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        micro_batch = self.preforward(micro_batch)
        # Pop non-tensor metadata that the model doesn't accept
        _channel_ids = micro_batch.pop("channel_ids", None)
        step_timer = None
        # dist.barrier()
        with self.model_fwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            step_timer = getattr(self, "_current_step_timer", None)
            with step_timer.measure("forward") if step_timer else contextlib.nullcontext():
                outputs: ModelOutput = self.model(**micro_batch, use_cache=False, step_timer=step_timer)

        # Restore channel_ids for postforward channel loss computation
        if _channel_ids is not None:
            micro_batch["channel_ids"] = _channel_ids

        loss: torch.Tensor
        loss_dict: Dict[str, torch.Tensor]
        loss, loss_dict = self.postforward(outputs, micro_batch)

        # Backward pass
        with self.model_bwd_context, set_batch_invariant_mode(self.args.train.enable_batch_invariant_mode):
            with step_timer.measure("backward") if step_timer else contextlib.nullcontext():
                loss.backward()

        del micro_batch
        return loss, loss_dict

    def model_reshard(self, micro_step: int, num_micro_steps: int):
        """Reshard model after backward pass."""
        args: VeOmniArguments = self.args
        if (
            args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
            and not args.train.accelerator.fsdp_config.reshard_after_backward
            and num_micro_steps > 1
        ):
            if micro_step == 0:
                self.model.set_reshard_after_backward(False)
            elif micro_step == num_micro_steps - 1:
                self.model.set_reshard_after_backward(True)


    def _sync_video_trained_num(self) -> bool:
        """Sync consumed/remaining data count across ranks and update video_trained_num.

        Returns True if training should stop (data exhausted on any rank).
        """
        args = self.args

        # Fake data mode: workers produce infinitely, no data_queue to track.
        # Use global_step directly as the progress counter.
        if getattr(args.train, "use_fake_data", False):
            self.video_trained_num = self.state.global_step
            self.state.video_trained_num = self.video_trained_num
            self.state.epoch = self.video_trained_num / max(self.init_data_size, 1)
            return

        if args.train.remote_dataloader and hasattr(self.train_dataloader, "remote_data_index"):
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
            if rank == 0:
                current_global_index = self.train_dataloader.remote_data_index.value
            else:
                current_global_index = 0
                
            data_tensor = torch.tensor([current_global_index], dtype=torch.long, device=self.device)
            
            if dist.is_initialized():
                dist.broadcast(data_tensor, src=0)
            
            self.video_trained_num = int(data_tensor.item())
            self.state.video_trained_num = self.video_trained_num
            self.state.epoch = self.video_trained_num / max(self.init_data_size, 1)
            
        else:
            if hasattr(self.train_dataloader, "samples_consumed"):
                # Ulysses PrefetchingPackedLoader path: samples_consumed is
                # the cumulative count of raw samples consumed on this DP rank.
                # All-reduce (sum) across DP ranks to get the global total.
                # Use dp_group to avoid double-counting SP ranks within the same DP group.
                consumed_per_rank = self.train_dataloader.samples_consumed
                consumed_tensor = torch.tensor(consumed_per_rank, dtype=torch.long, device=self.device)
                if dist.is_initialized():
                    ps = get_parallel_state()
                    dp_group = ps.dp_group if ps is not None else None
                    dist.all_reduce(consumed_tensor, op=dist.ReduceOp.SUM, group=dp_group)
                self.video_trained_num = int(consumed_tensor.item())
                self.state.video_trained_num = self.video_trained_num
                self.state.epoch = self.video_trained_num / max(self.init_data_size, 1)

            elif hasattr(self.train_dataloader, "data_queue"):
                remain_data = self.train_dataloader.data_queue.qsize()

                data_tensor_in = torch.tensor(remain_data, dtype=torch.long, device=self.device)
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                self._data_tensor_out = torch.zeros(world_size, dtype=torch.long, device=self.device)
                if dist.is_initialized():
                    dist.all_gather_into_tensor(self._data_tensor_out, data_tensor_in)
                else:
                    self._data_tensor_out[0] = data_tensor_in

                if get_parallel_state() is not None:
                    self._data_tensor_out = self._data_tensor_out // get_parallel_state().sp_size

                self.video_trained_num = self.init_data_size - int(self._data_tensor_out.sum().item())
                self.state.video_trained_num = self.video_trained_num
                self.state.epoch = self.video_trained_num / max(self.init_data_size, 1)

            else:
                remain_data = len(self.train_dataloader.data_list)
                logger.warning(
                    "dataloader has no attr data_queue or samples_consumed, defaulting to len(data_list)"
                )
                self.video_trained_num = self.init_data_size - remain_data
                self.state.video_trained_num = self.video_trained_num
                self.state.epoch = self.video_trained_num / max(self.init_data_size, 1)
 
    
    def _collect_modality_stats(self, micro_batches: List[Dict[str, Any]]) -> Dict[str, float]:
        """Collect per-step multimodal statistics for logging.
        """
        num_images = 0
        vision_tokens = 0
        num_audio_clips = 0
        audio_tokens = 0

        for mb in micro_batches:
            img_thw = mb.get("image_grid_thw")
            vid_thw = mb.get("video_grid_thw")
            if img_thw is not None and img_thw.numel() > 0:
                num_images += img_thw[:, 0].sum().item()
                vision_tokens += (img_thw[:, 0] * img_thw[:, 1] * img_thw[:, 2]).sum().item()
            if vid_thw is not None and vid_thw.numel() > 0:
                num_images += vid_thw[:, 0].sum().item()
                vision_tokens += (vid_thw[:, 0] * vid_thw[:, 1] * vid_thw[:, 2]).sum().item()

            audio_lens = mb.get("audio_features_lens")
            audio_feat = mb.get("audio_features")
            if audio_lens is not None and isinstance(audio_lens, torch.Tensor) and audio_lens.numel() > 0:
                num_audio_clips += audio_lens.shape[0]
                audio_tokens += audio_lens.sum().item()
            elif audio_feat is not None and audio_feat.numel() > 0:
                num_audio_clips += audio_feat.shape[0] if audio_feat.dim() == 3 else 1
                audio_tokens += audio_feat.shape[-1] * (audio_feat.shape[0] if audio_feat.dim() == 3 else 1)

        return {
            "mm_stats/num_images": num_images,
            "mm_stats/vision_tokens": vision_tokens,
            "mm_stats/num_audio_clips": num_audio_clips,
            "mm_stats/audio_tokens": audio_tokens,
        }

    def train_step(
        self,
        micro_batches: Any,
    ) -> Dict[str, float]:
        args = self.args

        # micro_batches: List[Dict[str, Any]] = next(data_iterator)
        if isinstance(micro_batches, dict):
            micro_batches = [micro_batches]

        self.on_step_begin(micro_batches=micro_batches)

        # Forward and backward for each micro batch
        synchronize()

        total_loss = 0.0
        total_loss_dict = defaultdict(int)

        # Multimodal stats
        total_loss_dict.update(self._collect_modality_stats(micro_batches))

        # token num for fixed_ce_loss in postforward
        self.micro_batches_token_len = count_loss_token(micro_batches)
        num_micro_steps = len(micro_batches)
        # forward and backward pass with gradient_accumulationsteps
        for micro_step, micro_batch in enumerate(micro_batches):
            self.model_reshard(micro_step, num_micro_steps)
            loss: torch.Tensor
            loss_dict: Dict[str, torch.Tensor]
            # token num for fixed_ce_loss in postforward
            self.micro_batch_token_len = count_loss_token(micro_batch)
            loss, loss_dict = self.forward_backward_step(micro_batch)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                total_loss_dict[k] += v.item() if isinstance(v, torch.Tensor) else v

        # Gradient clipping
        grad_norm = veomni_clip_grad_norm(self.model, args.train.optimizer.max_grad_norm)

        # Optimizer and scheduler step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self._sync_video_trained_num()

        warmup_steps = int(
            self.train_steps * args.train.optimizer.lr_warmup_ratio
        )
        if warmup_steps > 0 and self.video_trained_num < warmup_steps:
            self.lr_scheduler.step(epoch=self.video_trained_num)

        else:
            decay_ratio = getattr(args.train.optimizer, "lr_decay_ratio", 1.0)
            if self.video_trained_num / max(self.init_data_size, 1) <= decay_ratio:
                self.lr_scheduler.step(epoch=max(self.video_trained_num, warmup_steps))

        self.state.global_step += 1
        self.current_step += 1
        del micro_batches

        self.on_step_end(loss=total_loss, loss_dict=total_loss_dict, grad_norm=grad_norm)
    

        

    def destroy_distributed(self):
        # Clean up optimizer and lr scheduler
        del self.optimizer, self.lr_scheduler
        helper.empty_cache()

        dist.barrier()
        dist.destroy_process_group()

    def train(self):
        args: VeOmniArguments = self.args

        self.on_train_begin()

        self.train_dataloader.launch()
        self.state.max_steps = self.train_steps
        self.state.total_video_num = self.train_steps
        self.state.video_trained_num = 0
        self.state.num_train_epochs = args.train.num_train_epochs
        self.state.is_local_process_zero = (args.train.local_rank == 0)
        self.state.is_world_process_zero = (args.train.global_rank == 0)



        logger.info(
            f"Rank{args.train.local_rank} Start training. "
            f"Start step: {self.start_step}. "
            f"Train steps: {self.train_steps}. "
            f"Init data size per epoch: {self.init_data_size}. "
            f"Train epochs: {args.train.num_train_epochs}."
        )
        
        for epoch in range(args.train.num_train_epochs):
            if not self.train_dataloader.is_launched:
                self.train_dataloader.launch()
            data_iterator = iter(self.train_dataloader)
            self.current_epoch = epoch
            if hasattr(self.train_dataloader, "set_epoch"):
                self.train_dataloader.set_epoch(epoch)

            self.state.epoch = float(epoch)

            self.on_epoch_begin()
            start = self.start_step

            for step in range(start, self.init_data_size):
                self.current_step = step
                try:
                    micro_batches = next(data_iterator)
                    has_data = 1
                except:
                    import traceback
                    traceback.print_exc()
                    logger.info(f"epoch:{epoch} rank:{self.args.train.global_rank} Dataloader finished.")
                    micro_batches = None
                    has_data = 0

                if dist.is_initialized():
                    status_tensor = torch.tensor([has_data], dtype=torch.long, device=self.device)
                    dist.all_reduce(status_tensor, op=dist.ReduceOp.MIN)
                    should_continue = (status_tensor.item() == 1)
                else:
                    should_continue = (has_data == 1)
                
                if not should_continue:
                    logger.info(f"rank:{self.args.train.global_rank} Data exhausted on one or more ranks, stopping cleanly.")
                    break
                self.train_step(micro_batches)


            self.start_step = 0
            self.on_epoch_end()
            dist.barrier()
            self.train_dataloader.close()

            self.start_step = 0
            helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        self.on_train_end()
        synchronize()
        

        self.destroy_distributed()
