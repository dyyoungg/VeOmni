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

from typing import Optional

import torch
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


liger_kernel_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")


def fused_liger_kernel_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    weights = kwargs.pop("weights")
    hidden_states = kwargs.pop("hidden_states")
    return liger_kernel_cross_entropy(weights, hidden_states, labels), logits


# ---------------------------------------------------------------------------
# PATCH(veomni-big-chunk): cap Liger FLCE num_chunks at 4.
#
# liger chunk_size = next_pow2(BT / (V/H)); with small per-rank BT (packed
# seqs, e.g. BT~3.6k, V=151680, H=5120) this yields tiny 128-row chunks.
# EVERY chunk then pays a full [V, H] grad_weight accumulation
# (mm + .float() + add_ ~= 10GB HBM traffic), so 29 chunks cost ~300GB/step
# (~170ms) of pure dtype-cast traffic. Capping num_chunks at 4 cuts this ~7x
# while logits chunk memory stays <= 2.5GB (chunk <= 8192 rows).
# Enabled via VEOMNI_FLCE_BIG_CHUNK=1. No numerics change (same math, fewer
# accumulation steps).
# ---------------------------------------------------------------------------
import os as _os
import inspect as _inspect


def _install_flce_big_chunk_patch():
    import liger_kernel.ops.fused_linear_cross_entropy as _m

    if getattr(_m, "_veomni_big_chunk_patched", False):
        return
    src = _inspect.getsource(_m.fused_linear_cross_entropy_forward)
    old = "chunk_size = triton.next_power_of_2(triton.cdiv(BT, inc_factor))"
    new = (
        old + "\n"
        "    # PATCH(veomni-big-chunk): cap num_chunks at 4 - each chunk pays a\n"
        "    # full [V,H] grad_weight accumulation; tiny chunks explode HBM traffic.\n"
        "    chunk_size = max(chunk_size, min(triton.next_power_of_2(triton.cdiv(BT, 4)), 8192))"
    )
    if src.count(old) != 1:
        raise RuntimeError(f"FLCE big-chunk patch anchor not unique: count={src.count(old)}")
    _ns = dict(_m.__dict__)
    exec(compile(src.replace(old, new), "<flce_big_chunk>", "exec"), _ns)
    _m.fused_linear_cross_entropy_forward = _ns["fused_linear_cross_entropy_forward"]
    _m._veomni_big_chunk_patched = True
    print("[VeOmni] FLCE big-chunk patch installed (num_chunks<=4).", flush=True)


if _os.environ.get("VEOMNI_FLCE_BIG_CHUNK", "0") == "1":
    _install_flce_big_chunk_patch()