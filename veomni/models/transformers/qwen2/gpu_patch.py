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

import transformers.models.qwen2.modeling_qwen2 as hf_qwen2

from ....utils import logging
from ....utils.env import get_env
from ....utils.import_utils import is_liger_kernel_available


logger = logging.get_logger(__name__)


def apply_veomni_qwen2_gpu_patch():
    # ================================================================
    # PATCH: apply_rotary_pos_emb, Qwen2RMSNorm, Qwen2MLP
    # 1. Patch with Liger Kernel
    # ================================================================
    if is_liger_kernel_available() and get_env("VEOMNI_USE_LIGER_KERNEL") == "1":
        from liger_kernel.transformers.rms_norm import LigerRMSNorm
        from liger_kernel.transformers.rope import liger_rotary_pos_emb
        from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

        hf_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
        hf_qwen2.Qwen2RMSNorm = LigerRMSNorm
        hf_qwen2.Qwen2MLP = LigerSwiGLUMLP
        import os as _os
        if _os.environ.get("VEOMNI_FAST_RMS_NORM", "0") == "1":
            try:
                from veomni.ops.fast_rms_norm import FastRMSNorm

                _LigerRMSNorm = LigerRMSNorm

                class _AutoRMSNorm(FastRMSNorm):
                    def __init__(self, hidden_size, eps=1e-6):
                        if hidden_size != 5120:
                            # fall back for non-5120 instances (if any)
                            import liger_kernel.transformers.rms_norm as _lr
                            _lr.LigerRMSNorm.__init__(self, hidden_size, eps=eps)
                        else:
                            FastRMSNorm.__init__(self, hidden_size, eps=eps)

                hf_qwen2.Qwen2RMSNorm = _AutoRMSNorm
                logger.info_rank0("FastRMSNorm (custom HIP fwd+bwd) active for Qwen2.")
            except Exception as _e:
                logger.info_rank0(f"FastRMSNorm unavailable, keep LigerRMSNorm: {_e}")

        logger.info_rank0("Apply liger kernel to Qwen2.")
