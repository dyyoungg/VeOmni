"""
Runtime checkpoint tensor converter for ``LlavaQwen3MoeForCausalLM``.

Because ``LlavaQwen3MoeForCausalLM`` composes (not inherits)
``Qwen3MoeForCausalLM``, the per-expert HF-format converter registered on
the base class in ``veomni.models.transformers.qwen3_moe.__init__`` does
NOT reach this wrapper. We register our own factory that returns a
composite converter routing three possible on-disk key schemas to the
single live-model layout.

Live-model layout (v5 patchgen; see
``veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu``
line 260):
    model.layers.{L}.mlp.experts.gate_up_proj  [E, 2*I, H]  # cat([gate, up], dim=1)
    model.layers.{L}.mlp.experts.down_proj     [E, H, I]

Three on-disk schemas we care about:

1. **Legacy 3D-fused** (pre-transformers-v5 VeOmni builds; three separate
   ``nn.Parameter`` tensors, no expert index, no ``.weight`` suffix):
       model.layers.{L}.mlp.experts.gate_proj  [E, I, H]
       model.layers.{L}.mlp.experts.up_proj    [E, I, H]
       model.layers.{L}.mlp.experts.down_proj  [E, H, I]
   Handled by :class:`LlavaQwen3MoeLegacyExpertConverter` below —
   buffers gate/up per layer and merges into ``gate_up_proj``.

2. **Per-expert HF** (VeOmni :func:`export_weights` save path, or stock
   HF model repos; one linear-per-expert with ``.weight`` suffix):
       model.layers.{L}.mlp.experts.{j}.gate_proj.weight  [I, H]
       model.layers.{L}.mlp.experts.{j}.up_proj.weight    [I, H]
       model.layers.{L}.mlp.experts.{j}.down_proj.weight  [H, I]
   Handled by :class:`Qwen3MoeCheckpointTensorConverter` — buffers all
   experts per (layer, proj) and stacks then merges.

3. **v5 native fused** (HF ``save_pretrained`` on a live v5 model —
   state_dict emits ``nn.Parameter`` names directly):
       model.layers.{L}.mlp.experts.gate_up_proj  [E, 2*I, H]
       model.layers.{L}.mlp.experts.down_proj     [E, H, I]
   Neither pattern matches → the caller (``maybe_convert_checkpoint_tensor``)
   pass-throughs the tensor unchanged, which directly hits
   ``model.named_parameters()`` — no code needed here.

The three schemas are mutually exclusive on any single tensor key, so
routing by ``can_handle`` at the composite level is unambiguous.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import torch

from ....utils import logging
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor
from ...transformers.qwen3_moe.checkpoint_tensor_converter import Qwen3MoeCheckpointTensorConverter


logger = logging.get_logger(__name__)

# Legacy 3D-fused expert params under any MoE tower prefix, e.g.
# ``model.layers.12.mlp.experts.gate_proj``. No expert index, no
# ``.weight`` suffix -- these are ``nn.Parameter`` state_dict keys.
_LEGACY_EXPERT_PATTERN = re.compile(
    r"^(.+\.mlp)\.experts\.(gate_proj|up_proj|down_proj)$"
)


class LlavaQwen3MoeLegacyExpertConverter:
    """Convert legacy 3D-fused expert keys to v5 ``gate_up_proj`` layout.

    Buffers ``gate_proj`` / ``up_proj`` per layer prefix and emits a
    merged ``gate_up_proj`` once both arrive. ``down_proj`` passes
    through with an unchanged tensor (only the key needs canonicalizing).
    """

    def __init__(self) -> None:
        # {prefix: {"gate_proj"|"up_proj": tensor}}
        self._pending: Dict[str, Dict[str, torch.Tensor]] = {}

    def can_handle(self, name: str) -> bool:
        return bool(_LEGACY_EXPERT_PATTERN.match(name))

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        match = _LEGACY_EXPERT_PATTERN.match(name)
        if not match:
            return None

        prefix, proj_name = match.groups()

        if proj_name == "down_proj":
            # Same layout in old and new formats; converter contract
            # still requires we emit under a canonical name.
            return ConvertedCheckpointTensor(f"{prefix}.experts.down_proj", tensor)

        # gate_proj / up_proj — buffer for merging with the other.
        bucket = self._pending.setdefault(prefix, {})
        if proj_name in bucket:
            raise RuntimeError(
                f"LlavaQwen3Moe legacy converter: duplicate {proj_name!r} for prefix {prefix!r} "
                "(possibly reading the same shard twice)."
            )
        bucket[proj_name] = tensor

        if "gate_proj" in bucket and "up_proj" in bucket:
            gate = bucket.pop("gate_proj")
            up = bucket.pop("up_proj")
            if not bucket:
                del self._pending[prefix]
            # Match export_weights split: gate, up = chunk(2, dim=1) → cat along dim=1.
            merged = torch.cat([gate, up], dim=1)  # [E, 2*I, H]
            return ConvertedCheckpointTensor(f"{prefix}.experts.gate_up_proj", merged)

        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        if self._pending:
            unflushed = {prefix: list(bucket.keys()) for prefix, bucket in self._pending.items()}
            raise RuntimeError(
                "LlavaQwen3Moe legacy converter: incomplete checkpoint detected — "
                f"unflushed gate/up pairs (need both proj to merge into gate_up_proj): {unflushed}"
            )
        return []


class LlavaQwen3MoeCheckpointTensorConverter:
    """Composite converter routing between legacy 3D-fused and per-expert HF schemas.

    v5-native fused keys (``experts.gate_up_proj`` / ``experts.down_proj``) are
    NOT claimed by either sub-converter -- ``can_handle`` returns ``False``
    and the loader pass-throughs the tensor into ``named_parameters()``
    directly. This preserves zero-copy loading for the common case.
    """

    def __init__(self, num_experts: int):
        self._legacy = LlavaQwen3MoeLegacyExpertConverter()
        self._per_expert = Qwen3MoeCheckpointTensorConverter(num_experts=num_experts)

    def can_handle(self, name: str) -> bool:
        return self._legacy.can_handle(name) or self._per_expert.can_handle(name)

    def convert(self, name: str, tensor: "torch.Tensor") -> Optional[ConvertedCheckpointTensor]:
        # Dispatch by pattern -- schemas are mutually exclusive per key,
        # so at most one branch fires.
        if self._legacy.can_handle(name):
            return self._legacy.convert(name, tensor)
        if self._per_expert.can_handle(name):
            return self._per_expert.convert(name, tensor)
        return None

    def finalize(self) -> List[ConvertedCheckpointTensor]:
        # Concatenate flushes; each sub-converter raises on its own
        # incomplete state, so if we get here both are clean.
        return [*self._legacy.finalize(), *self._per_expert.finalize()]


def create_llava_qwen3moe_checkpoint_tensor_converter(model) -> LlavaQwen3MoeCheckpointTensorConverter:
    """Factory registered on ``LlavaQwen3MoeForCausalLM`` via
    ``_create_checkpoint_tensor_converter``.

    ``num_experts`` is required by :class:`Qwen3MoeCheckpointTensorConverter`
    to know when a per-expert (layer, proj) bucket is full. It's read from
    the wrapper's foundation config, which is set in the wrapper's
    ``__init__`` (see ``modeling_llava_qwen3moe_omni.py`` line ~99:
    ``self.num_experts = foundation_llm.num_experts``).
    """
    num_experts = getattr(model, "num_experts", None)
    if num_experts is None:
        # Fall back to reading from the foundation config directly.
        num_experts = model.foundation_config.num_experts
    return LlavaQwen3MoeCheckpointTensorConverter(num_experts=num_experts)
