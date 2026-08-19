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


from transformers import PretrainedConfig

from . import logging
from .device import get_device_name


logger = logging.get_logger(__name__)


def get_device_flops(unit="T"):
    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    device_name = get_device_name()
    flops = float("inf")  # INF flops for unkown gpu type
    if "H100" in device_name or "H800" in device_name or "H200" in device_name:
        flops = 989e12
    elif "A100" in device_name or "A800" in device_name:
        flops = 312e12
    elif "L40" in device_name:
        flops = 181.05e12
    elif "L20" in device_name:
        flops = 119.5e12
    elif "H20" in device_name:
        flops = 148e12
    elif "910B" in device_name:
        flops = 354e12
    elif "B200" in device_name:
        flops = 2250e12
    flops_unit = unit_convert(flops, unit)
    return flops_unit


class VeomniFlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = VeomniFlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(batch_seqlens, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        self.estimate_func = {
            "qwen2_vl": self._estimate_qwen2_vl_flops,
            # the only difference between Qwen2 and Qwen2.5 for counting flops is the window attention
            # used in the ViT for Qwen2.5VL which is considered in the _estimate_qwen2_vl_flops function.
            "qwen2_5_vl": self._estimate_qwen2_vl_flops,
            # qwen3_vl's vit uses full self attention while qwen2-vl/qwen2.5-vl uses window attention.
            "qwen3_vl": self._estimate_qwen3_vl_flops,
            "qwen3_vl_moe": self._estimate_qwen3_vl_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "qwen3_moe": self._estimate_qwen3_moe_flops,
            "llama": self._estimate_llama_flops,
            "qwen2": self._estimate_qwen2_flops,
            # qwen3_next
            "qwen3_next": self._estimate_qwen3_next_flops,
            # qwen3 reused _estimate_qwen2_flops func because the only model structure diff between qwen2 dense and qwen3 dense is that
            # qwen3 has additional RMSNorm layers for q and k.
            # RMSNorm layers have minimal impact at the MFU and can be ignored.
            "qwen3": self._estimate_qwen2_flops,
            "seed_oss": self._estimate_seed_flops,
            "qwen3_5": self._estimate_qwen3_5_family_flops,
            "qwen3_5_moe": self._estimate_qwen3_5_family_flops,
            "llavaqwen3moe_omni": self._estimate_llavaqwen3moeOmni_flops,
            "llavaqwen2_omni": self._estimate_llavaqwen2_omni_flops

        }

        self.config = config

    def _estimate_unknown_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        return 0

    def _estimate_seed_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts
        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts
        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank
        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers
        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_moe_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_num_expert = self.config.num_experts
        moe_topk = self.config.num_experts_per_tok

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_llama_flops(self, tokens_sum, batch_seqlens, delta_time):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # llama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen2_vl_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # vit flops
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens is not None:
            vit_flops = self._estimate_qwen_vit_flop(images_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved
    
 

    def _estimate_qwen3_vl_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
        hidden_size = self.config.text_config.hidden_size
        vocab_size = self.config.text_config.vocab_size
        num_hidden_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        num_attention_heads = self.config.text_config.num_attention_heads
        intermediate_size = self.config.text_config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # qwen3_vl uses deepstack to merge visual embeds and text embeds, but it has no tensor operation.

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # vit flops
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens is not None:
            vit_flops = self._estimate_qwen3_vit_flop(images_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved
    
    def _estimate_audio_flops(self, audio_seqlens, audio_config):
        """
        Estimate flops for Whisper encoder.
        audio_seqlens: list of mel feature lengths for each audio sample
        """
        # Whisper encoder 结构参数
        hidden_size = getattr(audio_config, "d_model", 1280)
        num_hidden_layers = getattr(audio_config, "encoder_layers", 32)
        num_attention_heads = getattr(audio_config, "encoder_attention_heads", 20)
        ffn_dim = getattr(audio_config, "encoder_ffn_dim", hidden_size * 4)

        head_dim = hidden_size // num_attention_heads

        # conv1d stem: 2层卷积将 mel 特征降采样
        # conv1(kernel=3) + conv2(kernel=3, stride=2)
        # flops = 2 * kernel_size * in_channels * out_channels * seq_len
        conv_flops = 0
        for audio_len in audio_seqlens:
            # conv1: mel_bins(80) -> hidden_size, kernel=3
            conv1_flops = 2 * 3 * 80 * hidden_size * audio_len
            # conv2: hidden_size -> hidden_size, kernel=3, stride=2, seq缩短一半
            conv2_flops = 2 * 3 * hidden_size * hidden_size * (audio_len // 2)
            conv_flops += conv1_flops + conv2_flops

        tokens_sum = sum(s // 2 for s in audio_seqlens)  # conv stride=2 后的实际长度

        # attention linear: Q K V O
        q_size = hidden_size
        k_size = hidden_size
        v_size = hidden_size
        attn_linear_N = hidden_size * (q_size + k_size + v_size + hidden_size)
        attn_linear_flops = 6 * attn_linear_N * num_hidden_layers * tokens_sum

        # FFN: fc1 + fc2 (Whisper 用 GELU，两个线性层)
        ffn_N = hidden_size * ffn_dim * 2
        ffn_flops = 6 * ffn_N * num_hidden_layers * tokens_sum

        # attention qkv flops（序列长度平方）
        seqlen_square_sum = sum((s // 2) * (s // 2) for s in audio_seqlens)
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        audio_flops = conv_flops + attn_linear_flops + ffn_flops + attn_qkv_flops

        return audio_flops
    
    def _estimate_llavaqwen2_omni_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        llm_config = getattr(self.config, "foundation_config", self.config)
        freeze_vit = kwargs.get("freeze_vit", True)
        freeze_audio = kwargs.get("freeze_audio", True)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        freeze_llm = kwargs.get("freeze_llm", False)
        hidden_size = llm_config.hidden_size
        vocab_size = llm_config.vocab_size
        num_hidden_layers = llm_config.num_hidden_layers
        num_key_value_heads = llm_config.num_key_value_heads
        num_attention_heads = llm_config.num_attention_heads
        intermediate_size = llm_config.intermediate_size

        head_dim = hidden_size // num_attention_heads
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # per layer params
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2

        # FLOPs系数
        if gradient_checkpointing:
            linear_factor = 6 if freeze_llm else 8
            attn_dot_factor = 16
        else:
            linear_factor = 4 if freeze_llm else 6
            attn_dot_factor = 12

        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers
        dense_N_flops = linear_factor * dense_N * tokens_sum
        # embedding & lm_head (不在GC范围内)
        emb_factor = 2 if freeze_llm else 6
        emb_flops = emb_factor * emd_and_lm_head_N * tokens_sum
        dense_N_flops += emb_flops

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = attn_dot_factor * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # vit flops
        images_seqlens = kwargs.get("images_seqlens", None)
        if images_seqlens is not None and getattr(self.config, "encoder_config", None) is not None:
            image_config = getattr(self.config.encoder_config, "image_config", None)
            if image_config is not None:
                freeze_projector = kwargs.get("freeze_projector", False)
                vit_flops = self._estimate_beebee_vit_flop(
                    images_seqlens, image_config,
                    freeze_vit=freeze_vit, freeze_projector=freeze_projector
                )
            else:
                vit_flops = 0
        else:
            vit_flops = 0

        # audio encoder flops（如果有）
        audio_seqlens = kwargs.get("audio_seqlens", None)
        if audio_seqlens is not None and getattr(self.config, "encoder_config", None) is not None:
            audio_config = getattr(self.config.encoder_config, "audio_config", None)
            if audio_config is not None:
                audio_flops = self._estimate_audio_flops(audio_seqlens, audio_config)
            else:
                audio_flops = 0
            audio_flops = audio_flops * (2.0 / 6.0) if freeze_audio else audio_flops

        else:
            audio_flops = 0

        flops_all = dense_N_flops + attn_qkv_flops + vit_flops + audio_flops
        flops_achieved = flops_all / delta_time / 1e12  # TFLOPs
        return flops_achieved
    
    def _estimate_beebee_vit_flop(self, images_seqlens, config, freeze_vit=False, freeze_projector=False):
        """
        Estimate ViT FLOPs for BeeBeeVL vision encoder (both Qwen2.5 and Qwen3.5 MoE variants).

        Architecture:
            PatchEmbed (Conv3d) -> ViT Blocks -> Merger (norm + reshape only, no linear)
            -> DynamicAvgPoolProjector (AdaptiveAvgPool2d + MLP)

        The merger has NO linear layer — just normalization + spatial token grouping.
        The projector MLP processes tokens AFTER pooling, so token count is greatly reduced.

        freeze_vit: ViT backbone is frozen (forward only, factor 2); projector still trains.
        freeze_projector: projector is also frozen (forward only, factor 2).
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth
        dim = config.hidden_size
        spatial_merge_size = config.spatial_merge_size
        head_dim = dim // num_heads

        # Determine if Qwen3.5 MoE ViT (GELU, factor 2) or Qwen2.5 ViT (SiLU/GLU, factor 3)
        is_qwen3_vit = "qwen3" in getattr(config, "model_type", "")
        mlp_hidden_dim = config.intermediate_size
        mlp_factor = 2 if is_qwen3_vit else 3

        # --- Patch embedding (Conv3d) ---
        in_channels = getattr(config, "in_channels", 3)
        temporal_patch_size = getattr(config, "temporal_patch_size", 2)
        patch_size = getattr(config, "patch_size", 14)
        patch_embed_N = dim * in_channels * temporal_patch_size * patch_size * patch_size

        # --- ViT blocks (process tokens_sum tokens) ---
        mlp_N = dim * mlp_hidden_dim * mlp_factor
        attn_linear_N = dim * (4 * dim)  # qkv (3*dim) + output proj (dim)
        vit_blocks_N = (mlp_N + attn_linear_N) * depth

        # --- Projector MLP (processes tokens AFTER merge + pool) ---
        # DynamicAvgPoolProjector: Linear(D, D) + GELU + Linear(D, output_size)
        # where D = dim * spatial_merge_size^2
        D = dim * (spatial_merge_size ** 2)
        output_size = getattr(config, "output_size", getattr(config, "out_hidden_size", D))
        projector_N = D * D + D * output_size  # = D * (D + output_size)

        # Token count after merge + pool:
        # After merge: tokens_sum / merge^2
        # After DynamicAvgPool with scale=image_downsample_size: further reduce by ~scale
        # get_adaptive_pool_size: r = 1/sqrt(scale), output ≈ input * r^2 = input / scale
        downsample_ratio = getattr(config, "image_downsample_size", 8)
        tokens_after_pool = tokens_sum / (spatial_merge_size ** 2 * downsample_ratio)

        # --- FLOPs factor: frozen = forward only (2x), trainable = fwd+bwd (6x) ---
        backbone_factor = 2 if freeze_vit else 6
        projector_factor = 2 if freeze_projector else 6

        # --- Dense FLOPs ---
        # ViT blocks + patch embed: process all tokens_sum
        backbone_flops = backbone_factor * (patch_embed_N + vit_blocks_N) * tokens_sum
        # Projector MLP: processes only post-pool tokens
        projector_flops = projector_factor * projector_N * tokens_after_pool

        # --- Attention QKV FLOPs ---
        # Determine full attention vs window attention layers
        if is_qwen3_vit:
            # Qwen3.5 MoE ViT uses full attention in all layers
            full_attn_layers = depth
            window_attn_layers = 0
        else:
            # Qwen2.5 ViT may use window attention in some layers
            fullatt_block_indexes = getattr(config, "fullatt_block_indexes", None)
            if fullatt_block_indexes is not None:
                full_attn_layers = len(fullatt_block_indexes)
            else:
                full_attn_layers = depth
            window_attn_layers = depth - full_attn_layers

        # attn compute: fwd = 4x, fwd+bwd = 12x
        attn_factor = 4 if freeze_vit else 12
        seqlen_square_sum = sum(s * s for s in images_seqlens)
        attn_qkv_flops = attn_factor * seqlen_square_sum * head_dim * num_heads * full_attn_layers

        if window_attn_layers > 0:
            window_size = getattr(config, "window_size", 112)
            window_attn_flops = attn_factor * tokens_sum * (window_size ** 2) * head_dim * num_heads
            attn_qkv_flops += window_attn_flops * window_attn_layers

        total_vit_flops = backbone_flops + projector_flops + attn_qkv_flops
        return total_vit_flops

    def _estimate_llavaqwen3moeOmni_flops(self, tokens_sum, batch_seqlens, delta_time, **kwargs):
        llm_config = getattr(self.config, "foundation_config", self.config)
        freeze_vit = kwargs.get("freeze_vit", True)
        freeze_audio = kwargs.get("freeze_audio", True)
        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        hidden_size = llm_config.hidden_size
        vocab_size = llm_config.vocab_size
        num_hidden_layers = llm_config.num_hidden_layers
        num_key_value_heads = llm_config.num_key_value_heads
        num_attention_heads = llm_config.num_attention_heads
        moe_intermediate_size = llm_config.moe_intermediate_size
        # MoE 相关
        num_experts_per_tok = getattr(llm_config, "num_experts_per_tok", 1)
        # Qwen3MoE 有 shared expert
        shared_expert_intermediate_size = getattr(llm_config, "shared_expert_intermediate_size", 0)

        # head_dim: Qwen3 MoE may have head_dim != hidden_size // num_attention_heads
        head_dim = getattr(llm_config, "head_dim", hidden_size // num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # attention linear 参数量（每层）
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

        # MoE FFN：只有被激活的 expert 参与计算
        # 每个 token 激活 num_experts_per_tok 个 expert
        moe_ffn_N_per_token = hidden_size * moe_intermediate_size * 3 * num_experts_per_tok
        # shared expert 每个 token 都要过
        shared_ffn_N = hidden_size * shared_expert_intermediate_size * 3 if shared_expert_intermediate_size > 0 else 0

        # embedding + lm_head
        emb_and_lm_head_N = vocab_size * hidden_size * 2

        # FLOPs系数:
        # - linear部分: frozen+GC = 6x (2fwd + 2recomp + 2act_grad), trainable+GC = 8x, trainable无GC = 6x
        # - attention dot: 有GC = 16x (4fwd + 4recomp + 8act_grad), 无GC = 12x (4fwd + 8bwd)
        freeze_llm = kwargs.get("freeze_llm", False)
        if gradient_checkpointing:
            linear_factor = 6 if freeze_llm else 8  # frozen无weight_grad省2x, GC加2x, 恰好=6
            attn_dot_factor = 16  # attention dot无权重, GC recomp始终增加4x
        else:
            linear_factor = 4 if freeze_llm else 6  # frozen: 2fwd+2act_grad=4
            attn_dot_factor = 12

        # attn linear
        attn_linear_flops = linear_factor * attn_linear_N * num_hidden_layers * tokens_sum
        # moe ffn（按激活参数量算，不是总参数量）
        moe_ffn_flops = linear_factor * (moe_ffn_N_per_token + shared_ffn_N) * num_hidden_layers * tokens_sum
        # embedding & lm_head (不在GC范围内，frozen时只有forward=2x)
        emb_factor = 2 if freeze_llm else 6
        emb_flops = emb_factor * emb_and_lm_head_N * tokens_sum

        dense_flops = attn_linear_flops + moe_ffn_flops + emb_flops

        # attention qkv flops（与序列长度平方相关）
        seqlen_square_sum = sum(s * s for s in batch_seqlens)
        attn_qkv_flops = attn_dot_factor * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # image encoder flops
        images_seqlens = kwargs.get("images_seqlens", None)
        if images_seqlens is not None and getattr(self.config, "encoder_config", None) is not None:
            image_config = getattr(self.config.encoder_config, "image_config", None)
            if image_config is not None:
                freeze_projector = kwargs.get("freeze_projector", False)
                vit_flops = self._estimate_beebee_vit_flop(
                    images_seqlens, image_config,
                    freeze_vit=freeze_vit, freeze_projector=freeze_projector
                )
            else:
                vit_flops = 0
        else:
            vit_flops = 0

        # audio encoder flops（如果有）
        audio_seqlens = kwargs.get("audio_seqlens", None)
        if audio_seqlens is not None and getattr(self.config, "encoder_config", None) is not None:
            audio_config = getattr(self.config.encoder_config, "audio_config", None)
            if audio_config is not None:
                audio_flops = self._estimate_audio_flops(audio_seqlens, audio_config)
            else:
                audio_flops = 0
            
            audio_flops = audio_flops * (2.0 / 6.0) if freeze_audio else audio_flops

        else:
            audio_flops = 0

        flops_all = dense_flops + attn_qkv_flops + vit_flops + audio_flops
        flops_achieved = flops_all / delta_time / 1e12  # TFLOPs
        # print(f" vit flops:{vit_flops}, LLM flops: {dense_flops+attn_qkv_flops}, audio flops: {audio_flops}")
        return flops_achieved

    def _estimate_qwen3_vl_moe_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
        hidden_size = self.config.text_config.hidden_size
        vocab_size = self.config.text_config.vocab_size
        moe_intermediate_size = self.config.text_config.moe_intermediate_size
        num_hidden_layers = self.config.text_config.num_hidden_layers
        num_key_value_heads = self.config.text_config.num_key_value_heads
        num_attention_heads = self.config.text_config.num_attention_heads
        moe_intermediate_size = self.config.text_config.moe_intermediate_size
        moe_num_expert = self.config.text_config.num_experts
        moe_topk = self.config.text_config.num_experts_per_tok
        head_dim = getattr(
            self.config.text_config,
            "head_dim",
            self.config.text_config.hidden_size // self.config.text_config.num_attention_heads,
        )
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim
        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * moe_N * tokens_sum
        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        # vit flops
        image_seqlens = kargs.get("image_seqlens", None)
        if image_seqlens is not None:
            vit_flops = self._estimate_qwen3_vit_flop(image_seqlens, self.config.vision_config)
        else:
            vit_flops = 0
        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_vit_flop(self, images_seqlens, config):
        """
        Estimate the FLOPS of the vision encoder for Qwen2 and Qwen2.5
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        dim = config.hidden_size
        mlp_hidden_dim = config.intermediate_size
        out_hidden_size = config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size

        head_dim = dim // num_heads

        # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
        patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
        # Qwen3 VL vision mlp does not use GLU, thus 2.
        mlp_N = dim * mlp_hidden_dim * 2
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

        # Qwen3 VL uses deep stack, one merger for every deepstack layer
        deepstack_merger_N = merger_N * len(config.deepstack_visual_indexes)
        # non-attn all_layer parm
        dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen3 VL, full attention is used in all vision layers.
        full_attn_layer_num = depth

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    def _estimate_qwen_vit_flop(self, images_seqlens, config):
        """
        Estimate the FLOPS of the vision encoder for Qwen2 and Qwen2.5
        """

        if config is None:
            return 0
        tokens_sum = sum(images_seqlens)

        num_heads = config.num_heads
        depth = config.depth

        # In Qwen2 VL and Qwen2.5VL, the parameters naming are different:
        #
        # Parameter                 | Qwen2 VL         | Qwen2.5 VL
        # --------------------------|------------------|------------------
        # ViT hidden dimension      | embed_dim        | hidden_size
        # ViT output dimension      | hidden_size      | out_hidden_size
        # ViT MLP intermediate dim  | embed_dim * mlp_ratio | intermediate_size
        #
        # See https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
        # and https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct/blob/main/config.json for an example.
        is_qwen2_vl = hasattr(config, "embed_dim")
        dim = config.embed_dim if is_qwen2_vl else config.hidden_size
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio) if is_qwen2_vl else config.intermediate_size
        out_hidden_size = config.hidden_size if is_qwen2_vl else config.out_hidden_size

        spatial_merge_size = config.spatial_merge_size
        head_dim = dim // num_heads

        # Qwen 2.5 VL uses SiLU, thus 3.
        mlp_N = dim * mlp_hidden_dim * (2 if is_qwen2_vl else 3)
        attn_linear_N = dim * (4 * dim)  # qkv and output proj
        patch_embed_and_merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (
            dim * (spatial_merge_size**2)
        )

        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * depth + patch_embed_and_merger_N

        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # In Qwen2.5 VL, windowed attention is used in some layers.
        full_attn_layer_num = config.depth if is_qwen2_vl else len(config.fullatt_block_indexes)
        window_attn_layer_num = config.depth - full_attn_layer_num

        # full attn layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in images_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

        # If window attention is used, add the window attention flops
        if window_attn_layer_num > 0:
            window_attn_compute_flops = 12 * tokens_sum * (config.window_size**2) * head_dim * num_heads
            attn_qkv_flops += window_attn_compute_flops * window_attn_layer_num

        vit_flops = dense_N_flops + attn_qkv_flops

        return vit_flops

    @staticmethod
    def _compute_hybrid_attn_params(config):
        """
        Compute hybrid attention (full + GatedDeltaNet) linear param count and layer info.

        Layers alternate between full attention and GatedDeltaNet (linear attention) in groups
        of `full_attention_interval` layers: (full_attention_interval - 1) linear layers followed
        by 1 full attention layer.

        Full attention (Qwen3_5Attention) projections:
            q_proj:  hidden_size -> num_attention_heads * head_dim  (output gate ignored, see note)
            k_proj:  hidden_size -> num_key_value_heads * head_dim
            v_proj:  hidden_size -> num_key_value_heads * head_dim
            o_proj:  num_attention_heads * head_dim -> hidden_size

        Note: q_proj actually outputs 2x (half query, half gate via sigmoid), but the gate
        contribution is ignored here for consistency with existing qwen3_next estimation.

        GatedDeltaNet (Qwen3_5GatedDeltaNet) projections:
            in_proj_qkv:  hidden_size -> 2 * linear_k_size + linear_v_size
            in_proj_z:    hidden_size -> linear_v_size          (output gate)
            in_proj_b:    hidden_size -> linear_num_value_heads (beta/gating scalar per head)
            in_proj_a:    hidden_size -> linear_num_value_heads (alpha/decay scalar per head)
            out_proj:     linear_v_size -> hidden_size
            conv1d:       depthwise, channels = 2 * linear_k_size + linear_v_size, kernel = conv_kernel_dim

        where:
            linear_k_size = linear_num_key_heads * linear_key_head_dim
            linear_v_size = linear_num_value_heads * linear_value_head_dim

        This only counts projection and conv1d parameter FLOPs. The GatedDeltaNet
        recurrence FLOPs are computed separately by _compute_gdn_recurrence_flops.
        """
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)

        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        full_attention_interval = config.full_attention_interval
        num_full_attn_layers = config.num_hidden_layers // full_attention_interval

        # Full attention: q_proj + k_proj + v_proj + o_proj
        full_attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

        # GatedDeltaNet linear projections and depthwise conv1d
        linear_k_size = config.linear_num_key_heads * config.linear_key_head_dim
        linear_v_size = config.linear_num_value_heads * config.linear_value_head_dim
        # in_proj_qkv                          + in_proj_z    + in_proj_b + in_proj_a              + out_proj
        # (2 * linear_k_size + linear_v_size)  + linear_v_size + linear_num_value_heads * 2        + linear_v_size
        linear_attn_size = 2 * linear_k_size + 2 * linear_v_size + 2 * config.linear_num_value_heads + linear_v_size
        # depthwise conv1d: each of (2 * linear_k_size + linear_v_size) channels has its own kernel
        conv_N = config.linear_conv_kernel_dim * (2 * linear_k_size + linear_v_size)
        linear_attn_linear_N = hidden_size * linear_attn_size + conv_N

        # Each group has 1 full attention layer + (full_attention_interval - 1) GatedDeltaNet layers
        num_linear_attn_per_group = full_attention_interval - 1
        attn_linear_N = (full_attn_linear_N + num_linear_attn_per_group * linear_attn_linear_N) * num_full_attn_layers

        return attn_linear_N, num_full_attn_layers, head_dim, num_attention_heads

    @staticmethod
    def _compute_gdn_recurrence_flops(config, tokens_sum, num_full_attn_layers):
        """
        Compute FLOPs for the GatedDeltaNet recurrence across all GDN layers.

        The recurrent form of GatedDeltaNet (ref: https://kexue.fm/archives/11033, eq.17/18):

            S_t = gamma_t * S_{t-1} + eta_t * (v_t - S_{t-1} @ k_t) @ k_t^T
            o_t = S_t @ q_t

        where S_t is the state matrix of shape (linear_value_head_dim, linear_key_head_dim)
        per value head.

        Note: in practice, training uses the chunked implementation (chunk_gated_delta_rule)
        which reorganizes the computation into chunk-level matrix multiplications for better
        hardware utilization. However, chunking is purely an implementation optimization that
        does not change the total arithmetic — it computes the same result as the recurrent
        form. We therefore use the recurrent form as the theoretical FLOPs baseline.

        Per step per head, the dominant ops (forward) are:
            S_{t-1} @ k_t        (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
            (...) @ k_t^T        (outer product, (d_v,)⊗(d_k,)=(d_v,d_k)):  d_v * d_k FLOPs
            o_t = S_t @ q_t      (mat-vec, (d_v,d_k)@(d_k,)=(d_v,)):  2 * d_v * d_k FLOPs
        where d_v = linear_value_head_dim, d_k = linear_key_head_dim.

        Following the same convention as quadratic attention (Q@K + attn@V):
            fwd: (2 + 1 + 2) * d_v * d_k = 5 * d_v * d_k per step per head
            fwd + bwd (3x): 15 * d_v * d_k per step per head
        """
        num_gdn_layers = config.num_hidden_layers - num_full_attn_layers
        return (
            15
            * config.linear_key_head_dim
            * config.linear_value_head_dim
            * config.linear_num_value_heads
            * tokens_sum
            * num_gdn_layers
        )

    def _estimate_qwen3_next_flops(self, tokens_sum, batch_seqlens, delta_time):
        """
        Estimate the FLOPS of the Qwen3 Next model.
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers

        # hybrid attention params
        attn_linear_N, num_full_attn_layers, head_dim, num_attention_heads = self._compute_hybrid_attn_params(
            self.config
        )

        # moe per layer parm
        # TopkGate layer and gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
        moe_gata_N = hidden_size * self.config.num_experts
        moe_sharedexpertmlp_N = hidden_size * self.config.shared_expert_intermediate_size * 3
        moe_expertmlp_N = hidden_size * self.config.moe_intermediate_size * self.config.num_experts_per_tok * 3
        moe_N = (moe_gata_N + moe_expertmlp_N + moe_sharedexpertmlp_N) * num_hidden_layers

        # lm head param
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * (moe_N + attn_linear_N + emd_and_lm_head_N) * tokens_sum
        # attn all_layer & all_token fwd & bwd flops, only count full attention layers
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        # GatedDeltaNet recurrence flops (state update + query, for all GDN layers)
        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(self.config, tokens_sum, num_full_attn_layers)

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_qwen3_5_family_flops(self, tokens_sum, batch_seqlens, delta_time, **kargs):
        """
        Estimate the FLOPS of the Qwen3.5 model family (dense/MoE MLP + hybrid attention + ViT).

        Handles both Qwen3.5 (dense) and Qwen3.5-MoE by checking for MoE-specific config
        attributes. Both variants share hybrid attention and ViT; only the MLP differs.

        Text model (from text_config):
            Dense MLP per layer (SwiGLU, 3 projections):
                gate_proj:  hidden_size -> intermediate_size
                up_proj:    hidden_size -> intermediate_size
                down_proj:  intermediate_size -> hidden_size

            MoE per layer (when num_experts is present):
                TopkGate router:   hidden_size -> num_experts
                Routed experts (top-k activated, each SwiGLU):
                    gate_proj:  hidden_size -> moe_intermediate_size
                    up_proj:    hidden_size -> moe_intermediate_size
                    down_proj:  moe_intermediate_size -> hidden_size
                    -> 3 projections * num_experts_per_tok active experts
                Shared expert (always active, SwiGLU):
                    gate_proj:  hidden_size -> shared_expert_intermediate_size
                    up_proj:    hidden_size -> shared_expert_intermediate_size
                    down_proj:  shared_expert_intermediate_size -> hidden_size

            Hybrid attention: see _compute_hybrid_attn_params docstring.

            Embeddings + LM head:
                embed_tokens:  vocab_size -> hidden_size
                lm_head:       hidden_size -> vocab_size

        Quadratic attention FLOPs (only full attention layers):
            Per layer: 2 * seq_len^2 * head_dim * num_attention_heads (Q@K + attn@V)
            fwd + bwd (3x) -> 6x total -> coefficient 12

        Vision encoder: delegates to _estimate_qwen3_vit_flop.
        """
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size
        num_hidden_layers = text_config.num_hidden_layers

        # hybrid attention linear projection params (full + GatedDeltaNet)
        attn_linear_N, num_full_attn_layers, head_dim, num_attention_heads = self._compute_hybrid_attn_params(
            text_config
        )

        # MLP params: MoE or dense depending on config
        is_moe = hasattr(text_config, "num_experts")
        if is_moe:
            # MoE per layer: router gate + routed expert MLPs (top-k) + shared expert MLP
            moe_gata_N = hidden_size * text_config.num_experts
            moe_expertmlp_N = hidden_size * text_config.moe_intermediate_size * text_config.num_experts_per_tok * 3
            moe_sharedexpertmlp_N = hidden_size * text_config.shared_expert_intermediate_size * 3
            mlp_N = (moe_gata_N + moe_expertmlp_N + moe_sharedexpertmlp_N) * num_hidden_layers
        else:
            # dense MLP per layer: gate_proj + up_proj + down_proj (SwiGLU)
            mlp_N = hidden_size * text_config.intermediate_size * 3 * num_hidden_layers

        # embed_tokens + lm_head
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # linear projection flops: 6 (fwd + bwd) * params * tokens
        dense_N_flops = 6 * (mlp_N + attn_linear_N + emd_and_lm_head_N) * tokens_sum

        # quadratic attention flops (Q@K and attn@V), only for full attention layers
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_full_attn_layers

        # GatedDeltaNet recurrence flops (state update + query, for all GDN layers)
        gdn_recurrence_flops = self._compute_gdn_recurrence_flops(text_config, tokens_sum, num_full_attn_layers)

        # vit flops (Qwen3-VL ViT)
        images_seqlens = kargs.get("images_seqlens", None)
        if images_seqlens is not None:
            vit_flops = self._estimate_qwen3_vit_flop(images_seqlens, self.config.vision_config)
        else:
            vit_flops = 0

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops + gdn_recurrence_flops + vit_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time, **kwargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_unknown_flops)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time, **kwargs)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops
