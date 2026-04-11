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

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers.modeling_outputs import ModelOutput

from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import gather_outputs
from ..utils import logging
from ..utils.constants import IGNORE_INDEX, MODALITY
from ..utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids, valid_seqlens_from_cu_seqlens


logger = logging.get_logger(__name__)


def add_flash_attention_kwargs_from_position_ids(
    batch: Dict[str, "torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Calculate and add Flash Attention kwargs (cu_seq_lens and max_length) from position_ids.

    Pass down already computed cu_seq_lens and max_length as the HF transformers
    FlashAttentionKwargs naming so that it can be used without recomputation every layer.
    HF model code would handle the pass down of those kwargs for us.
    Note that the recomputation would cause host->device sync which hurts performance and
    stability due to CPU instability.

    Args:
        batch: The batch dictionary containing position_ids. Will be modified in-place to add
               cu_seq_lens_q, cu_seq_lens_k, max_length_q, and max_length_k.

    Returns:
        Tuple of (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k) for additional use.
    """
    position_ids = batch["position_ids"]
    if position_ids.dim() == 3:  # bs, dim, seq_len
        position_ids = position_ids[:, 0, :]
    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    batch["cu_seq_lens_q"] = cu_seq_lens_q
    batch["cu_seq_lens_k"] = cu_seq_lens_k
    batch["max_length_q"] = max_length_q
    batch["max_length_k"] = max_length_k

    return cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k


@dataclass
class DataCollateInfo:
    pack_dim: int = field(
        default=0,
        metadata={"help": "Dim to pack in batch. Default is 0. If -1, pack in last dim and unsqueeze(0)"},
    )
    sp_slice: bool = field(
        default=False,
        metadata={"help": "Whether to sp slice in batch. Default is False"},
    )
    sp_pad_value: int = field(
        default=None,
        metadata={"help": "sp_pad value of a sequence in batch. Not pad if None. Default is None"},
    )
    sp_pad_scale: int = field(
        default=1,
        metadata={"help": "sp_pad scale of a sequence in batch. Default is 1"},
    )

    def __post_init__(self):
        assert self.pack_dim is not None, "pack_dim must be specified"
        if self.sp_slice:
            assert self.sp_pad_value is not None and self.sp_pad_scale is not None, (
                "sp_pad_value and sp_pad_scale must be specified when sp_slice is True"
            )

        assert (self.sp_pad_value is None) == (self.sp_pad_scale is None), (
            "sp_pad_value and sp_pad_scale must be specified together or None"
        )


# pack_dim, sp_slice, sp_pad_value, sp_pad_scale
DEFAULT_DATA_COLLATE_INFO: Dict[str, DataCollateInfo] = {
    "input_ids": DataCollateInfo(-1, True, 0, 1),
    "labels": DataCollateInfo(-1, True, IGNORE_INDEX, 1),
    "attention_mask": DataCollateInfo(-1, False, 1, 1),
    "position_ids": DataCollateInfo(-1, False, 0, 1),
    "pixel_values": DataCollateInfo(0, True, 0, 4),
    "pixel_values_videos": DataCollateInfo(0, True, 0, 4),
    "image_mask": DataCollateInfo(-1, False, 0, 1),
    "video_mask": DataCollateInfo(-1, False, 0, 1),
    "image_grid_hw": DataCollateInfo(0, False, None, None),
    "image_grid_thw": DataCollateInfo(0, False, None, None),
    "video_grid_thw": DataCollateInfo(0, False, None, None),
}


@dataclass
class DataCollator(ABC):
    """
    Used in dataloader as a collate_fn.
    """

    @abstractmethod
    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        """
        Converts a list of features to batched tensor dict.
        """
        ...


@dataclass
class NoopDataCollator(DataCollator):
    """
    Data collator with no operation, used when collating in preforward.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> List[Dict[str, "torch.Tensor"]]:
        return features


@dataclass
class UnpackDataCollator(DataCollator):
    """
    Data collator to unpack examples, used in dynamic batch dataloader.
    """

    def __call__(self, features: Sequence[Dict[str, "torch.Tensor"]]) -> Dict[str, "torch.Tensor"]:
        return features[0]


@dataclass
class MakeMicroBatchCollator(DataCollator):
    """
    Data collator to build micro batches, used in mapping dataloader.
    """

    num_micro_batch: int
    internal_data_collator: "DataCollator"

    def __call__(self, features: Sequence[Tuple[Dict[str, "torch.Tensor"]]]) -> List[Dict[str, "torch.Tensor"]]:
        micro_batch_size = len(features) // self.num_micro_batch
        for i in range(len(features)):
            features[i] = features[i][0]  # 1-to-N inverse transform

        micro_batches = []
        for i in range(0, len(features), micro_batch_size):
            micro_batches.append(self.internal_data_collator(features[i : i + micro_batch_size]))

        return micro_batches


@dataclass
class PrecomputePositionIDsCollator(DataCollator):
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for feature in features:
            if "position_ids" not in feature:
                # default position_ids is 0 ~ seq_len - 1 for text models
                feature["position_ids"] = torch.arange(feature["input_ids"].size(-1), dtype=torch.int64)
        return features


@dataclass
class PackingCollator(DataCollator):
    collate_infos: Dict[str, DataCollateInfo] = field(default_factory=lambda: DEFAULT_DATA_COLLATE_INFO.copy())
    pad_to_length: int = False
    seq_classification: bool = (
        False  # whether the training task is sequence classification, if true, do not mask boundary labels
    )

    def __post_init__(self):
        self.sp_enabled = get_parallel_state().sp_enabled

    def pad_feature_to_length(
        self,
        feature: Union[torch.Tensor, List[torch.Tensor]],
        dim: int = -1,
        pad_value: int = 0,
        pad_size: int = 0,
    ) -> torch.Tensor:
        pad_shape = list(feature.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=feature.dtype, device=feature.device)
        return torch.cat((feature, pad), dim=dim)

    def pad_batch_to_length(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_len = batch["input_ids"].shape[-1]
        assert seq_len <= self.pad_to_length, "pad_to_length must be >= packed sequence length."

        pad_len = self.pad_to_length - seq_len
        if pad_len == 0:
            return batch

        keys_to_pad = ["input_ids", "attention_mask", "labels", "position_ids"]
        for key in keys_to_pad:
            if key in batch:
                batch[key] = self.pad_feature_to_length(
                    batch[key],
                    dim=self.collate_infos[key].pack_dim,
                    pad_value=self.collate_infos[key].sp_pad_value,
                    pad_size=pad_len,
                )
        return batch

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = defaultdict(list)
        for feature in features:
            for key in feature.keys():
                batch[key].append(feature[key])

        for key in batch.keys():
            collate_info: DataCollateInfo = self.collate_infos.get(key, None)
            if collate_info is None:
                try:
                    if key.split("_")[0] in MODALITY:
                        batch[key] = torch.cat(batch[key], dim=0)
                    else:
                        batch[key] = default_collate(batch[key])
                except Exception:
                    # use List of tensor, for example: num, height, width, c in different resolution
                    pass
            else:
                pack_dim = collate_info.pack_dim

                # first token of packed sequence must be IGNORE_INDEX
                if key == "labels" and not self.seq_classification:
                    for i in range(1, len(batch[key])):
                        batch[key][i][0] = IGNORE_INDEX

                batch[key] = torch.cat(batch[key], dim=pack_dim)
                if pack_dim == -1:
                    batch[key] = batch[key].unsqueeze(0)

        if self.pad_to_length:
            batch = self.pad_batch_to_length(batch)

        if not self.sp_enabled:
            add_flash_attention_kwargs_from_position_ids(batch)
        return batch


@dataclass
class SequenceParallelCollator(DataCollator):
    collate_infos: Dict[str, DataCollateInfo] = field(default_factory=lambda: DEFAULT_DATA_COLLATE_INFO.copy())
    seq_classification: bool = (
        False  # whether the training task is sequence classification, if true, do not shift labels
    )

    def __post_init__(self):
        self.sp_size = get_parallel_state().sp_size
        self.sp_rank = get_parallel_state().sp_rank

    def sp_slice(self, key: str, feature: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if isinstance(feature, list):
            assert dim == 0, f"Only support dim=0 for {key} as it is a List"
            seq_length = len(feature)
            sp_chunk_size = seq_length // self.sp_size
            return feature[self.sp_rank * sp_chunk_size : (self.sp_rank + 1) * sp_chunk_size]
        else:
            seq_length = feature.size(dim)
            sp_chunk_size = seq_length // self.sp_size
            return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self,
        key: str,
        feature: Union[torch.Tensor, List[torch.Tensor]],
        dim: int = -1,
        pad_value: int = 0,
        pad_scale: int = 1,
    ) -> torch.Tensor:
        if isinstance(feature, List):
            assert dim == 0, f"Only support dim=0 for {key} as {key} is a List of Tensor"
            seq_length = len(feature)
        else:
            seq_length = feature.size(dim)

        scale_sp_size = self.sp_size * pad_scale
        sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
        pad_size = sp_chunk_size * scale_sp_size - seq_length
        if pad_size == 0:
            return feature

        if isinstance(feature, List):
            # if feature is uncatable, pad pad_size num feature[-1] to the List
            feature += [feature[-1]] * pad_size
            return feature
        else:
            pad_shape = list(feature.shape)
            pad_shape[dim] = pad_size
            pad = torch.full(pad_shape, fill_value=pad_value, dtype=feature.dtype, device=feature.device)
            return torch.cat((feature, pad), dim=dim)

    def __call__(self, batch: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        if not self.seq_classification:
            # shift labels
            labels = batch["labels"][..., 1:].contiguous()
            labels = F.pad(labels, (0, 1), "constant", IGNORE_INDEX)
            batch["labels"] = labels

        for key in batch.keys():
            collate_info: DataCollateInfo = self.collate_infos.get(key, None)
            if collate_info is None:
                continue
            pack_dim = collate_info.pack_dim
            sp_slice = collate_info.sp_slice
            sp_pad_value = collate_info.sp_pad_value
            sp_pad_scale = collate_info.sp_pad_scale
            if sp_pad_value is not None:
                # sp padding
                batch[key] = self.sp_padding(
                    key,
                    batch[key],
                    dim=pack_dim,
                    pad_value=sp_pad_value,
                    pad_scale=sp_pad_scale,
                )

            if sp_slice and key != "position_ids":  # position_ids should be sp sliced after precompute fa kwargs
                # sp slice
                batch[key] = self.sp_slice(key, batch[key], dim=pack_dim)

        add_flash_attention_kwargs_from_position_ids(batch)

        batch["position_ids"] = self.sp_slice(
            "position_ids", batch["position_ids"], dim=self.collate_infos["position_ids"].pack_dim
        )

        return batch


@dataclass
class MainCollator(DataCollator):
    data_collate_info: Dict[str, Union[DataCollateInfo, tuple, Dict]] = field(default_factory=lambda: {})
    pad_to_length: bool = False
    seq_classification: bool = False

    """
    Data collator pipeline with a unified collate info.

    Args:
        data_collate_info:
            User config to override the default collate info.
        pad_to_length:
            Whether to pad sequence to a fixed length. Default is False.
        seq_classification:
            If True, sequence classification task. Default is False.
    """

    def __post_init__(self):
        self.preforward_pipeline = []
        self.collate_infos: Dict[str, DataCollateInfo] = {}

        full_info = DEFAULT_DATA_COLLATE_INFO.copy()
        full_info.update(self.data_collate_info)

        for name, params in full_info.items():
            if isinstance(params, DataCollateInfo):
                self.collate_infos[name] = params
            elif isinstance(params, dict):
                self.collate_infos[name] = DataCollateInfo(**params)
            elif isinstance(params, tuple):
                self.collate_infos[name] = DataCollateInfo(*params)

        """attention_mask always pad 1
        VeOmni sp slice `input_ids` & `labels` while keeps the full sequence of `attention_mask`. This leads to wrong behavior of `create_causal_mask` in transformers.
        `create_causal_mask` will slice the `attention_mask` to `attention_mask[-len(input_ids):]`.
        refer to https://github.com/huggingface/transformers/blob/bdc85cb85c8772d37aa29ce447860b44d7fad6ef/src/transformers/masking_utils.py#L770
        So VeOmni make sure attention_mask is all_ones when using flash_attn, and precalculate the position_ids & cu_seqlens & max_seqlens.
        """
        assert self.collate_infos["attention_mask"].sp_pad_value == 1

        self.preforward_pipeline.append(PrecomputePositionIDsCollator())
        self.preforward_pipeline.append(
            PackingCollator(
                collate_infos=self.collate_infos,
                pad_to_length=self.pad_to_length,
                seq_classification=self.seq_classification,
            )
        )
        if get_parallel_state().sp_enabled:
            self.preforward_pipeline.append(
                SequenceParallelCollator(collate_infos=self.collate_infos, seq_classification=self.seq_classification)
            )
        logger.info_rank0(self.log_collate_infos())

    def __call__(self, micro_batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        for preforward_func in self.preforward_pipeline:
            micro_batch = preforward_func(micro_batch)
        return micro_batch

    def log_collate_infos(self) -> None:
        sample_info = next(iter(self.collate_infos.values()))
        fields = list(asdict(sample_info).keys())

        header = ["name"] + fields

        row_format = "{:<25}" + "{:<18}" * len(fields)

        log_str = ""
        log_str += "\n" + "=" * (25 + 18 * len(fields)) + "\n"
        log_str += "Main Collate Configuration\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        log_str += row_format.format(*header) + "\n"
        log_str += "-" * (25 + 18 * len(fields)) + "\n"

        for name, info in self.collate_infos.items():
            row_data = [name] + [str(getattr(info, f)) for f in fields]
            log_str += row_format.format(*row_data) + "\n"

        log_str += "=" * (25 + 18 * len(fields)) + "\n"
        return log_str


@dataclass
class PostCollator(DataCollator):
    def __init__(self):
        self.postforward_pipeline = []
        self.compute_seqlens_func = SeqlensComputePostCollator()
        self.postforward_pipeline.append(PackingPostCollator())

    def __call__(self, outputs: ModelOutput, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = self.compute_seqlens_func(micro_batch)
        for postforward_func in self.postforward_pipeline:
            outputs = postforward_func(outputs, seq_lens)
        return outputs


@dataclass
class SeqlensComputePostCollator(DataCollator):
    def __call__(self, micro_batch: Dict[str, torch.Tensor]):
        seq_lens = valid_seqlens_from_cu_seqlens(micro_batch["cu_seq_lens_q"]).tolist()
        return seq_lens


@dataclass
class PackingPostCollator(DataCollator):
    def __call__(self, outputs: ModelOutput, seq_lens):
        logits = outputs.logits
        if get_parallel_state().sp_enabled:
            logits = gather_outputs(logits, gather_dim=0, group=get_parallel_state().sp_group)
            logits = logits[: sum(seq_lens)]  # remove sp padding
        logits_list = logits.split(seq_lens, dim=0)
        outputs.logits = logits_list
        return outputs



@dataclass
class UlysessOmniDataSharderCollator:
    """
    Collator for packing + Ulysses Sequence Parallel training.
 
    Input contract (batch_size == 1, packing mode)
    -----------------------------------------------
    The single element in `features` is the packed sample dict produced by
    MultimodalPacker._pack_and_yield().  Expected keys:
 
        input_ids          : [T]          – packed token ids
        labels             : [T]          – packed labels
        sample_lens        : [N]          – per-sample sequence lengths (sum == T)
        pixel_values       : [P, D]       – (optional) image patch features
        image_grid_thw     : [M, 3]       – (optional) image grid info
        pixel_values_video : [V, D]       – (optional) video patch features
        video_grid_thw     : [K, 3]       – (optional) video grid info
        audio_features     : [A, F]       – (optional) audio features
        audio_features_lens: [Na]         – (optional) audio feature lengths
        attention_mask     : [T]          – (optional) 1-d attention mask
 
    Output keys (FSDP2 flash-attention packing convention)
    -------------------------------------------------------
        input_ids          : [1, T_pad//sp]
        labels             : [1, T_pad//sp]
        pixel_values       : [P_pad//sp, D]  (image + video merged, SP-sliced)
        image_grid_thw     : [M+K, 3]        (image + video merged, NOT sliced)
        audio_features     : [A, F]          (NOT sliced)
        audio_features_lens: [Na]            (NOT sliced)
        seq_lens           : [N]             – per-sample lengths (full, pre-slice)
        cu_seq_lens_q      : [N+1] or [N+2]  – cumulative seqlens for flash-attn;
                                               N+2 when SP padding adds a dummy segment
        cu_seq_lens_k      : same as cu_seq_lens_q
        max_length_q       : int             – max real per-sample length (pre-slice)
        max_length_k       : int             – same as max_length_q
        attention_mask     : [1, T_pad//sp]  – (optional)
 
    Notes on cu_seq_lens
    --------------------
    flash-attn varlen kernels require cu_seqlens[-1] == actual tensor length fed
    to the kernel (i.e. T_pad, the SP-padded length, NOT the original T).
 
    When SP padding is needed (T_pad > T), we append one extra dummy segment
    [T, T_pad] to cu_seqlens instead of stretching the last real segment, so
    the dummy padding tokens are causally isolated and never attend to real ones.
    Their labels are IGNORE_INDEX so they contribute nothing to the loss.
    """
 
    pad_token_id: int = 0
    ignore_index: int = IGNORE_INDEX
 
    # dim along which each key is SP-padded then SP-sliced
    sp_slice_features: Dict[str, int] = field(
        default_factory=lambda: {
            "input_ids":      -1,   # [1, T]   → last dim
            "labels":         -1,   # [1, T]   → last dim
            "attention_mask": -1,   # [1, T]   → last dim
            "pixel_values":    0,   # [P, D]   → first dim
        }
    )
 
    padding_features: Dict[str, Any] = field(
        default_factory=lambda: {
            "input_ids":      0,            # overridden by pad_token_id in __post_init__
            "labels":         IGNORE_INDEX,
            "attention_mask": 0,
            "pixel_values":   0,
            "audio_features": 0.0,
        }
    )
 
    # pixel_values patches come in groups of 4 (temporal window), so their
    # padded length must be divisible by sp_size * 4.
    # Token sequences have no extra grouping constraint (scale = 1).
    padding_scale: Dict[str, int] = field(
        default_factory=lambda: {
            "pixel_values": 4,
        }
    )
 
    def __post_init__(self):
        try:
            ps = get_parallel_state()
            self.sp_size = getattr(ps, "sp_size", 1)
            self.sp_rank = getattr(ps, "sp_rank", 0)
        except Exception:
            self.sp_size = 1
            self.sp_rank = 0
 
        self.padding_features["input_ids"] = self.pad_token_id
 
    # ── SP helpers ────────────────────────────────────────────────────────────
 
    def _padded_length(self, original: int, dim_key: str) -> int:
        """Return the SP-aligned length for a given feature key."""
        if self.sp_size <= 1:
            return original
        scale       = self.padding_scale.get(dim_key, 1)
        target_unit = self.sp_size * scale
        return (original + target_unit - 1) // target_unit * target_unit
 
    def _sp_padding(
        self,
        tensor: torch.Tensor,
        dim: int,
        pad_value: float,
        dim_key: str,
    ) -> torch.Tensor:
        """Pad `tensor` along `dim` to the SP-aligned length."""
        if self.sp_size <= 1:
            return tensor
 
        original   = tensor.size(dim)
        target_len = self._padded_length(original, dim_key)
        pad_size   = target_len - original
 
        if pad_size <= 0:
            return tensor
 
        pad_shape      = list(tensor.shape)
        pad_shape[dim] = pad_size
        pad_tensor     = torch.full(
            pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device
        )
        return torch.cat([tensor, pad_tensor], dim=dim)
 
    def _sp_slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Extract this rank's contiguous chunk along `dim`."""
        if self.sp_size <= 1:
            return tensor
 
        total      = tensor.size(dim)
        chunk_size = total // self.sp_size
        start      = self.sp_rank * chunk_size
        return tensor.narrow(dim, start, chunk_size).contiguous()
 
    # ── Main collation ────────────────────────────────────────────────────────
 
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert len(features) == 1, (
            f"UlysessOmniDataSharderCollator only supports batch_size=1 "
            f"(packing mode), got {len(features)}."
        )
        raw   = features[0]
        batch: Dict[str, Any] = {}
 
        def _to_tensor(val, dtype=None):
            if isinstance(val, torch.Tensor):
                return val if dtype is None else val.to(dtype)
            if isinstance(val, list):
                if val and isinstance(val[0], torch.Tensor):
                    return torch.cat(val, dim=0)
                return torch.tensor(val, dtype=dtype)
            raise ValueError(f"Unsupported type: {type(val)}")
 
        # ── 1. Text tokens ────────────────────────────────────────────────────
        if "input_ids" in raw:
            batch["input_ids"] = _to_tensor(raw["input_ids"], torch.long).unsqueeze(0)   # [1, T]
 
        if "labels" in raw:
            batch["labels"] = _to_tensor(raw["labels"], torch.long).unsqueeze(0)         # [1, T]
 
        if "attention_mask" in raw:
            batch["attention_mask"] = _to_tensor(raw["attention_mask"]).unsqueeze(0)     # [1, T]
 
        # ── 2. Per-sample lengths ─────────────────────────────────────────────
        if "sample_lens" in raw:
            batch["seq_lens"] = _to_tensor(raw["sample_lens"], torch.int32)              # [N]
 
        # ── 3. Label shift + inter-sample boundary masking ────────────────────
      
        if "labels" in batch and "seq_lens" in batch:
            labels = batch["labels"]                    # [1, T]
 
            labels = labels[..., 1:].contiguous()
            labels = F.pad(labels, (0, 1), "constant", self.ignore_index)
 
            # mask the last token of every sample except the final one
            # (the final sample's last position is already IGNORE from the shift pad)
            # cu         = F.pad(batch["seq_lens"].cumsum(0), (1, 0), value=0)
            # boundaries = cu[1:-1]                       # inter-sample boundaries, [N-1]
            # if boundaries.numel() > 0:
            #     labels[0, boundaries - 1] = self.ignore_index
 
            batch["labels"] = labels
 
        # ── 4. Flash-attention metadata (BEFORE SP padding/slice) ─────────────
        #
        # The flash-attn varlen kernel requires:
        #     cu_seqlens[-1] == length of the tensor actually passed in
        #
        # After SP padding the tensor length becomes T_padded, so cu_seqlens
        # must reflect that.  We append a dummy segment [T_original, T_padded]
        # rather than stretching the last real segment, so padding tokens are
        # causally isolated.  Their labels are IGNORE_INDEX → no loss contribution.
        #
        # All SP ranks derive T_padded from the same deterministic formula, so
        # cu_seqlens is identical across the SP group (required for correctness).
        #
        if "seq_lens" in batch:
            seq_lens   = batch["seq_lens"]                              # [N], int32
            T_original = int(seq_lens.sum().item())
            # tokens have no extra scale (scale=1); use _padded_length for consistency
            T_padded   = self._padded_length(T_original, "input_ids")
            max_seqlen = int(seq_lens.max().item())
 
            cu_seqlens = F.pad(
                seq_lens.cumsum(dim=0).to(torch.int32), (1, 0), value=0
            )                                                           # [N+1]
 
            if T_padded > T_original:
                cu_seqlens = F.pad(cu_seqlens, (0, 1), value=T_padded) # [N+2]
 
            batch["cu_seq_lens_q"] = cu_seqlens
            batch["cu_seq_lens_k"] = cu_seqlens
            batch["max_length_q"]  = max_seqlen     # plain int, not tensor
            batch["max_length_k"]  = max_seqlen
 
        # ── 5. Vision modalities ──────────────────────────────────────────────
        merged_pv: List[torch.Tensor] = []
        for key in ("pixel_values", "pixel_values_video"):
            if key in raw and raw[key] is not None:
                merged_pv.append(_to_tensor(raw[key]))
        if merged_pv:
            batch["pixel_values"] = torch.cat(merged_pv, dim=0)        # [P_img+P_vid, D]
 
        merged_thw: List[torch.Tensor] = []
        for key in ("image_grid_thw", "video_grid_thw"):
            if key in raw and raw[key] is not None:
                merged_thw.append(_to_tensor(raw[key]))
        if merged_thw:
            batch["image_grid_thw"] = torch.cat(merged_thw, dim=0)     # [M+K, 3]
 
        for key in ("audio_features", "audio_features_lens"):
            if key in raw and raw[key] is not None:
                batch[key] = _to_tensor(raw[key])
 
        # ── 7. SP padding → SP slice ──────────────────────────────────────────
        #
        # Excluded from SP treatment (must be identical on every rank):
        #   cu_seq_lens_q/k, max_length_q/k, seq_lens, image_grid_thw, audio_*
        #
        for key, dim in self.sp_slice_features.items():
            if key not in batch:
                continue
            pad_val = self.padding_features.get(key, 0)
            batch[key] = self._sp_padding(
                batch[key], dim=dim, pad_value=pad_val, dim_key=key
            )
 
        for key, dim in self.sp_slice_features.items():
            if key not in batch:
                continue
            batch[key] = self._sp_slice(batch[key], dim=dim)
 
        return batch