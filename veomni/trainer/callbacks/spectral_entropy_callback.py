# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Low-frequency singular-value entropy monitoring for trainable weights."""

import math
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from veomni.optim.muon import _fsdp_all2all_submesh, _shard_row_sizes
from veomni.utils.device import stream_synchronize
from veomni.utils.logging import get_logger

from .base import Callback, TrainerState

if TYPE_CHECKING:
    from ..base import BaseTrainer


logger = get_logger(__name__)

_PARAMETER_TYPES = ("projector", "attention_qo", "attention_kv", "router", "expert", "dense")
_PARAMETER_SCOPES = ("image_encoder", "llm")


def singular_value_entropy(weight: torch.Tensor, max_slices: int = 4) -> torch.Tensor:
    """Return normalized singular-value entropy for a 2D or batched 2D tensor.

    For a matrix ``W``, the probabilities are ``p_i = s_i**2 / sum(s**2)``,
    where ``s_i`` are singular values.  For a 3D MoE weight, the final two
    dimensions are treated as matrices and the result is averaged over at most
    ``max_slices`` matrices.
    """
    if weight.ndim < 2:
        raise ValueError(f"singular_value_entropy expects ndim >= 2, got {weight.ndim}")

    matrices = weight if weight.ndim == 2 else weight.reshape(-1, weight.shape[-2], weight.shape[-1])
    if weight.ndim > 2 and max_slices > 0 and matrices.shape[0] > max_slices:
        indices = torch.linspace(
            0, matrices.shape[0] - 1, max_slices, device=matrices.device, dtype=torch.float32
        ).round().to(dtype=torch.long)
        matrices = matrices.index_select(0, indices)

    matrices = matrices.float()
    singular_values = torch.linalg.svdvals(matrices)
    energy = singular_values.square()
    total_energy = energy.sum(dim=-1, keepdim=True)
    probabilities = energy / total_energy.clamp_min(torch.finfo(energy.dtype).tiny)
    log_probabilities = probabilities.clamp_min(torch.finfo(energy.dtype).tiny).log()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    normalizer = math.log(singular_values.shape[-1])
    if normalizer == 0.0:
        return torch.zeros((), dtype=entropy.dtype, device=entropy.device)
    return (entropy / normalizer).mean()


_LAYER_TOKEN = re.compile(r"(?:^|\.)(layers?|blocks?|h)\.(\d+)(?:\.|$)")


def parameter_layer_name(name: str) -> str:
    """Map a parameter name to a stable, compact layer group."""
    match = _LAYER_TOKEN.search(name)
    if match is None:
        return "global"
    prefix = name[: match.start()].rstrip(".").split(".")
    stem = "_".join(prefix[-2:]) if prefix else "model"
    return f"{stem}_{match.group(1)}_{match.group(2)}"


def parameter_layer_index(name: str) -> int | None:
    """Return the zero-based transformer layer index in a parameter name."""
    match = _LAYER_TOKEN.search(name)
    return int(match.group(2)) if match is not None else None


def parameter_type(name: str) -> str:
    """Classify projector, attention, MoE and dense parameter names."""
    lower = name.lower()
    # Image projectors are small but directly control the visual-to-LLM
    # interface, so keep them visible as their own metric type. Audio
    # projectors are filtered by ``parameter_scope`` before measurement.
    if re.search(r"(?:^|[._])(?:mm|image|vision|audio)_projector(?:[._]|$)", lower) or (
        "image_encoder" in lower and re.search(r"(?:^|[._])projector(?:[._]|$)", lower)
    ):
        return "projector"
    if re.search(r"(?:^|[._])experts?(?:[._]|$)|(?:^|[._])expert[0-9]+(?:[._]|$)", lower):
        return "expert"
    if re.search(r"(?:router|routing|expert_gate)(?:[._]|$)", lower) or re.search(
        r"(?:^|[._])mlp[._]gate(?:[._]|$)", lower
    ):
        return "router"
    if re.search(r"(?:^|[._])(?:k|v|kv)(?:_a|_b)?_proj(?:[._]|$)", lower):
        return "attention_kv"
    if re.search(r"(?:^|[._])(?:key|value|kv_proj)(?:[._]|$)", lower):
        return "attention_kv"
    if re.search(r"(?:^|[._])(?:q|o)(?:_a|_b)?_proj(?:[._]|$)", lower):
        return "attention_qo"
    if re.search(r"(?:^|[._])(?:query|out_proj|o_proj)(?:[._]|$)", lower):
        return "attention_qo"
    return "dense"


def parameter_scope(name: str) -> str:
    """Separate image, audio and language-model parameters."""
    lower = name.lower()
    audio_tokens = ("audio_encoder", "audio_tower", "audio_projector")
    if any(token in lower for token in audio_tokens):
        return "audio_encoder"
    image_tokens = (
        "image_encoder",
        "vision_tower",
        "vision_encoder",
        "vision_model",
        "visual",
        "image_projector",
        "mm_projector",
    )
    return "image_encoder" if any(token in lower for token in image_tokens) else "llm"


def _parameter_dtensor(parameter: torch.Tensor) -> DTensor | None:
    if isinstance(parameter, DTensor):
        return parameter
    data = getattr(parameter, "data", None)
    return data if isinstance(data, DTensor) else None


def _is_row_sharded(dtensor: DTensor) -> bool:
    return sum(
        getattr(placement, "dim", None) == 0
        for placement in dtensor.placements
        if hasattr(placement, "dim")
    ) == 1


def _reduce_local_shard_scalar(value: torch.Tensor, dtensor: DTensor) -> torch.Tensor:
    """Average a scalar over mesh dimensions that shard the DTensor."""
    for mesh_dim, placement in enumerate(dtensor.placements):
        if not isinstance(placement, Shard):
            continue
        group = dtensor.device_mesh.get_group(mesh_dim)
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
        value.div_(dtensor.device_mesh.size(mesh_dim))
    return value


def _owner_all2all_entropy(
    entries: list[tuple[str, torch.Tensor]],
    mesh,
    max_slices: int,
) -> dict[str, float]:
    """Gather one full matrix per owner, compute SVD, then send scalars back."""
    world = mesh.size(0)
    if len(entries) > world:
        raise ValueError(f"Expected at most {world} entries, got {len(entries)}")

    owner_rank = int(mesh.get_coordinate()[0])
    group = mesh.get_group(0)
    local_tensors = []
    for _, parameter in entries:
        dtensor = _parameter_dtensor(parameter)
        local_tensors.append(dtensor.to_local().detach() if dtensor is not None else parameter.detach())
    device = local_tensors[0].device if local_tensors else torch.device("cuda")
    dtype = local_tensors[0].dtype if local_tensors else torch.float32

    owner_entry = entries[owner_rank] if owner_rank < len(entries) else None
    if owner_entry is None:
        owner_rows, owner_cols, owner_dtype = [0] * world, 0, dtype
    else:
        owner_tensor = _parameter_dtensor(owner_entry[1])
        owner_local = owner_tensor.to_local() if owner_tensor is not None else owner_entry[1]
        owner_rows = _shard_row_sizes(int(owner_entry[1].shape[0]), world)
        owner_cols, owner_dtype = int(owner_entry[1].shape[1]), owner_local.dtype

    # Flatten the per-destination matrices and use explicit element split
    # sizes.  The tensor-list all_to_all API is fragile for this bucket: only
    # the first few destinations have real parameters and the remaining
    # destinations are dummy slots, while every owner has a different matrix
    # width.  all_to_all_single expresses the same source/destination pairing
    # without relying on NCCL's handling of uneven tensor-list entries.
    send_chunks = [
        local_tensors[index].contiguous().reshape(-1)
        if index < len(local_tensors)
        else torch.ones(1, device=device, dtype=dtype)
        for index in range(world)
    ]
    input_split_sizes = [int(chunk.numel()) for chunk in send_chunks]
    send_flat = torch.cat(send_chunks, dim=0)
    if owner_entry is not None:
        output_split_sizes = [int(rows * owner_cols) for rows in owner_rows]
        recv_shapes = [(rows, owner_cols) for rows in owner_rows]
    else:
        output_split_sizes = [1] * world
        recv_shapes = [(1, 1)] * world
    recv_flat = torch.empty(
        sum(output_split_sizes), device=device, dtype=owner_dtype
    )
    dist.all_to_all_single(
        recv_flat,
        send_flat,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    if device.type != "cpu":
        # NCCL's synchronous API orders the current stream but may return
        # before GPU completion. Wait on the host before entering cuSolver
        # so local gather work cannot overlap its host-side driver calls.
        stream_synchronize()
    recv_list = []
    offset = 0
    for split_size, shape in zip(output_split_sizes, recv_shapes, strict=True):
        recv_list.append(recv_flat[offset : offset + split_size].reshape(shape))
        offset += split_size
    if owner_entry is not None:
        full_matrix = torch.cat(recv_list, dim=0)
        with torch.no_grad():
            owner_value = singular_value_entropy(full_matrix, max_slices=max_slices).to(torch.float32)
        del full_matrix
    else:
        owner_value = torch.zeros((), device=device, dtype=torch.float32)

    # Each owner sends its scalar to every rank. Receivers index the source rank,
    # which is also the parameter position owned by that rank.
    scalar_send = owner_value.reshape(1).expand(world).contiguous()
    if owner_entry is None:
        scalar_send.zero_()
    scalar_recv = torch.empty(world, device=device, dtype=torch.float32)
    dist.all_to_all_single(scalar_recv, scalar_send, group=group)
    return {entries[index][0]: scalar_recv[index].item() for index in range(len(entries))}


class SpectralEntropyMonitorCallback(Callback):
    """Periodically measure singular-value entropy of a deterministic parameter sample.

    FSDP2 row-sharded 2D parameters use two owner all-to-all operations: the
    first gathers rows to one owner per parameter, and the second sends scalar
    entropy values back. Large 3D MoE tensors use local shards by default and
    all-reduce their scalar estimates.
    """

    def __init__(self, trainer: "BaseTrainer") -> None:
        super().__init__(trainer)
        self._parameters: list[tuple[str, torch.Tensor]] = []
        self._enabled = False

        args = self.trainer.args
        self.interval = int(getattr(args.train, "spectral_entropy_monitor_interval", 0))
        self.max_slices = int(getattr(args.train, "spectral_entropy_monitor_max_slices", 4))
        self.log_per_param = bool(getattr(args.train, "spectral_entropy_monitor_log_per_param", True))
        self.gather_3d = bool(getattr(args.train, "spectral_entropy_monitor_gather_3d", False))
        layers = getattr(args.train, "spectral_entropy_monitor_layers", [])
        if not isinstance(layers, (list, tuple)) or any(type(layer) is not int or layer < 0 for layer in layers):
            raise ValueError("spectral_entropy_monitor_layers must be a list of nonnegative integer layer indices")
        self.requested_layers = tuple(dict.fromkeys(layers))

    @property
    def _rank(self) -> int:
        return int(getattr(self.trainer.args.train, "global_rank", os.environ.get("RANK", 0)))

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        if self.interval <= 0:
            return
        if not self.requested_layers:
            logger.warning_rank0(
                "Spectral entropy monitor is enabled but spectral_entropy_monitor_layers is empty; "
                "no parameters will be sampled."
            )
            return
        if self.max_slices <= 0:
            raise ValueError("spectral_entropy_monitor_max_slices must be positive when monitoring is enabled")

        module = getattr(self.trainer.model, "unwrapped_module", self.trainer.model)
        module = getattr(module, "module", module)
        candidates = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
            and parameter.ndim in (2, 3)
            and parameter_scope(name) != "audio_encoder"
        ]
        candidates.sort(key=lambda item: item[0])

        if self.requested_layers:
            requested = set(self.requested_layers)
            candidates = [
                candidate
                for candidate in candidates
                if parameter_layer_index(candidate[0]) in requested
                or (
                    parameter_scope(candidate[0]) == "image_encoder"
                    and parameter_type(candidate[0]) == "projector"
                )
            ]
            found = {parameter_layer_index(name) for name, _ in candidates}
            missing = sorted(requested - found)
            if missing:
                logger.warning_rank0(
                    "Spectral entropy monitor requested layers with no matching trainable parameters: "
                    f"{missing}"
                )
        self._parameters = candidates
        self._enabled = bool(self._parameters)
        if not self._enabled:
            logger.warning_rank0("Spectral entropy monitor enabled, but no trainable 2D/3D parameters were found.")
            return

        logger.info_rank0(
            "Spectral entropy monitor enabled: "
            f"interval={self.interval}, sampled_params={len(self._parameters)}, "
            f"max_slices={self.max_slices}, gather_3d={self.gather_3d}, "
            f"layers={list(self.requested_layers)}"
        )
        logger.info_rank0("Spectral entropy parameters: " + ", ".join(name for name, _ in self._parameters))

    def _measure_non_a2a(self, name: str, parameter: torch.Tensor) -> dict[str, float]:
        dtensor = _parameter_dtensor(parameter)
        if dtensor is not None and parameter.ndim == 3 and not self.gather_3d:
            # Full MoE expert tensors can be several GB. Compute a local-shard
            # estimate and average the scalar across ranks.
            value = singular_value_entropy(dtensor.to_local().detach(), max_slices=self.max_slices)
            if dist.is_available() and dist.is_initialized():
                _reduce_local_shard_scalar(value, dtensor)
            return {name: value.item()} if self._rank == 0 else {}

        tensor = dtensor.full_tensor() if dtensor is not None else parameter.data
        value = singular_value_entropy(tensor.detach(), max_slices=self.max_slices).item()
        return {name: value} if self._rank == 0 else {}

    def _measure_parameters(self) -> dict[str, float]:
        """Measure all selected parameters with owner all-to-all where possible."""
        values: dict[str, float] = {}
        a2a_groups: dict[tuple[Any, torch.dtype], list[tuple[str, torch.Tensor]]] = {}

        for name, parameter in self._parameters:
            dtensor = _parameter_dtensor(parameter)
            mesh = (
                _fsdp_all2all_submesh(dtensor)
                if dtensor is not None and parameter.ndim == 2 and _is_row_sharded(dtensor)
                else None
            )
            if mesh is None:
                values.update(self._measure_non_a2a(name, parameter))
            else:
                key = (mesh, dtensor.to_local().dtype)
                a2a_groups.setdefault(key, []).append((name, parameter))

        for (mesh, _dtype), group_entries in a2a_groups.items():
            world = mesh.size(0)
            for start in range(0, len(group_entries), world):
                chunk = group_entries[start : start + world]
                values.update(
                    _owner_all2all_entropy(
                        chunk, mesh, self.max_slices
                    )
                )
        return values

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if not self._enabled or state.global_step % self.interval != 0:
            return

        values = self._measure_parameters()

        if self._rank != 0 or not values:
            return

        value_tensor = torch.tensor(list(values.values()), dtype=torch.float32)
        metrics = {
            "spectral_entropy/overall/mean": value_tensor.mean().item(),
            "spectral_entropy/overall/min": value_tensor.min().item(),
            "spectral_entropy/overall/max": value_tensor.max().item(),
        }
        by_scope: dict[str, list[float]] = defaultdict(list)
        by_scope_type: dict[tuple[str, str], list[float]] = defaultdict(list)
        for name, value in values.items():
            kind = parameter_type(name)
            scope = parameter_scope(name)
            by_scope[scope].append(value)
            by_scope_type[scope, kind].append(value)
        for scope in _PARAMETER_SCOPES:
            scoped_values = by_scope[scope]
            if scoped_values:
                prefix = f"spectral_entropy/{scope}"
                metrics[f"{prefix}/mean"] = sum(scoped_values) / len(scoped_values)
                metrics[f"{prefix}/min"] = min(scoped_values)
                metrics[f"{prefix}/max"] = max(scoped_values)
            for kind in _PARAMETER_TYPES:
                scoped_values = by_scope_type[scope, kind]
                if scoped_values:
                    metrics[f"spectral_entropy/{scope}/type/{kind}"] = sum(scoped_values) / len(scoped_values)
        if self.log_per_param:
            for name, value in values.items():
                safe_name = name.replace("/", "_")
                scope = parameter_scope(name)
                metrics[f"spectral_entropy/{scope}/param/{safe_name}"] = value

        self.trainer.step_env_metrics.update(metrics)
        logger.info_rank0(
            f"[step {state.global_step}] spectral_entropy: "
            + ", ".join(
                f"{scope}={metrics[f'spectral_entropy/{scope}/mean']:.4f}"
                for scope in _PARAMETER_SCOPES
                if f"spectral_entropy/{scope}/mean" in metrics
            )
        )

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        self._parameters.clear()
        self._enabled = False
