from unittest import SkipTest

import torch
import torch.nn.functional as F

from veomni.models.custom.llava_qwen3moe.projector_frame_attention import (
    FrameVarlenAttentionProjector,
    flash_attn_varlen_func,
    get_adaptive_pool_size,
)


def _require_cuda():
    if not torch.cuda.is_available() or flash_attn_varlen_func is None:
        raise SkipTest("Requires CUDA and flash-attn.")


def _adaptive_avgpool_reference(features, grids, merge_sizes, ratios):
    outputs = []
    seq_len = []
    offset = 0
    for (time, raw_height, raw_width), merge_size, ratio in zip(grids.tolist(), merge_sizes, ratios):
        height = raw_height // merge_size
        width = raw_width // merge_size
        length = time * height * width
        grid_features = features[offset : offset + length].view(time, height, width, -1)
        output_height, output_width = get_adaptive_pool_size(height, width, ratio)
        pooled = F.adaptive_avg_pool2d(
            grid_features.permute(0, 3, 1, 2), (output_height, output_width)
        )
        outputs.append(pooled.permute(0, 2, 3, 1).reshape(-1, features.shape[-1]))
        seq_len.extend([output_height * output_width] * time)
        offset += length
    if outputs:
        return torch.cat(outputs), seq_len
    return features.new_empty((0, features.shape[-1])), seq_len


def test_zero_initialized_attention_matches_dynamic_average_pooling():
    _require_cuda()
    torch.manual_seed(0)
    grids = torch.tensor([[2, 10, 14], [1, 12, 8]])
    merge_sizes = [2, 2]
    ratios = [4.0, 12.0]
    merged_lengths = [
        time * (height // merge_size) * (width // merge_size)
        for (time, height, width), merge_size in zip(grids.tolist(), merge_sizes)
    ]
    features = torch.randn(sum(merged_lengths), 16)
    projector = FrameVarlenAttentionProjector(
        16, 16, downsample_ratio=4, num_attention_heads=4
    )

    projector = projector.cuda().bfloat16()
    features = features.cuda().bfloat16()
    enhanced, seq_len, aux = projector.enhance_tokens(
        features,
        grids,
        merge_size=merge_sizes,
        downsample_ratios=ratios,
    )
    expected, expected_seq_len = _adaptive_avgpool_reference(
        features, grids, merge_sizes, ratios
    )

    torch.testing.assert_close(enhanced, expected)
    torch.testing.assert_close(aux.base_tokens, expected)
    assert torch.count_nonzero(aux.residual_tokens) == 0
    assert seq_len == expected_seq_len


def test_frame_varlen_boundaries_follow_packed_grids():
    _require_cuda()
    grids = torch.tensor([[2, 8, 12], [1, 12, 8]])
    merge_sizes = [2, 2]
    ratios = [4.0, 3.0]
    feature_count = sum(
        time * (height // merge_size) * (width // merge_size)
        for (time, height, width), merge_size in zip(grids.tolist(), merge_sizes)
    )
    projector = FrameVarlenAttentionProjector(
        8, 12, downsample_ratio=4, num_attention_heads=2
    )

    projector = projector.cuda().bfloat16()
    output, seq_len, aux = projector(
        torch.randn(feature_count, 8, device="cuda", dtype=torch.bfloat16),
        grids,
        merge_size=merge_sizes,
        downsample_ratios=ratios,
        return_attention_aux=True,
    )

    first_output = get_adaptive_pool_size(4, 6, 4.0)
    second_output = get_adaptive_pool_size(6, 4, 3.0)
    expected_query_lengths = [first_output[0] * first_output[1]] * 2 + [
        second_output[0] * second_output[1]
    ]
    expected_key_value_lengths = [4 * 6, 4 * 6, 6 * 4]

    assert seq_len == expected_query_lengths
    assert aux.query_lengths == expected_query_lengths
    assert aux.key_value_lengths == expected_key_value_lengths
    assert aux.cu_seqlens_q.tolist() == [0, 6, 12, 18]
    assert aux.cu_seqlens_kv.tolist() == [0, 24, 48, 72]
    assert output.shape == (sum(expected_query_lengths), 12)


def test_frames_do_not_exchange_information():
    _require_cuda()
    torch.manual_seed(1)
    grids = torch.tensor([[2, 8, 8]])
    features = torch.randn(32, 8)
    projector = FrameVarlenAttentionProjector(
        8, 8, downsample_ratio=4, num_attention_heads=2
    )
    projector = projector.cuda().bfloat16()
    features = features.detach().cuda().bfloat16().requires_grad_(True)
    with torch.no_grad():
        projector.attention_out.weight.copy_(torch.eye(8))

    first_output, _, _ = projector.enhance_tokens(features, grids)
    changed_features = features.clone()
    changed_features[16:] += 100
    second_output, _, _ = projector.enhance_tokens(changed_features, grids)

    tokens_per_frame = get_adaptive_pool_size(4, 4, 4.0)
    tokens_per_frame = tokens_per_frame[0] * tokens_per_frame[1]
    torch.testing.assert_close(
        first_output[:tokens_per_frame], second_output[:tokens_per_frame]
    )
    assert not torch.allclose(
        first_output[tokens_per_frame:], second_output[tokens_per_frame:]
    )


def test_attention_residual_and_qkv_receive_gradients():
    _require_cuda()
    torch.manual_seed(2)
    grids = torch.tensor([[1, 8, 8]])
    features = torch.randn(16, 8, requires_grad=True)
    projector = FrameVarlenAttentionProjector(
        8, 6, downsample_ratio=4, num_attention_heads=2
    )
    projector = projector.cuda().bfloat16()
    features = features.detach().cuda().bfloat16().requires_grad_(True)
    with torch.no_grad():
        projector.attention_out.weight.copy_(torch.eye(8) * 0.01)

    output, _ = projector(features, grids)
    coefficients = torch.arange(output.numel(), dtype=output.dtype, device=output.device).view_as(output)
    (output * coefficients).sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert projector.attention_out.weight.grad is not None
    assert projector.attention_out.weight.grad.abs().sum() > 0
    assert projector.query_proj.weight.grad is not None
    assert projector.query_proj.weight.grad.abs().sum() > 0
    assert projector.key_proj.weight.grad is not None
    assert projector.key_proj.weight.grad.abs().sum() > 0
    assert projector.value_proj.weight.grad is not None
    assert projector.value_proj.weight.grad.abs().sum() > 0


def test_meta_materialization_restores_zero_residual_projection():
    with torch.device("meta"):
        projector = FrameVarlenAttentionProjector(
            8, 8, downsample_ratio=4, num_attention_heads=2
        )

    assert projector.attention_out.weight.is_meta
    projector.to_empty(device="cpu")

    assert torch.count_nonzero(projector.attention_out.weight) == 0
    assert projector.attention_out.weight._veomni_fsdp_shard_dim == 1


def test_empty_packed_visual_input_is_supported():
    _require_cuda()
    projector = FrameVarlenAttentionProjector(
        8, 6, downsample_ratio=4, num_attention_heads=2
    )

    projector = projector.cuda().bfloat16()
    features = torch.empty((0, 8), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output, seq_len, aux = projector(
        features,
        torch.empty((0, 3), dtype=torch.long),
        return_attention_aux=True,
    )
    output.sum().backward()

    assert output.shape == (0, 6)
    assert seq_len == []
    assert aux.cu_seqlens_q.tolist() == [0]
    assert aux.cu_seqlens_kv.tolist() == [0]
    assert features.grad is not None
    for name, parameter in projector.named_parameters():
        assert parameter.grad is not None, f"Missing empty-input gradient for {name}"


def test_invalid_feature_length_is_rejected():
    projector = FrameVarlenAttentionProjector(
        8, 8, downsample_ratio=4, num_attention_heads=2
    )
    grids = torch.tensor([[1, 8, 8]])

    try:
        projector(torch.randn(15, 8), grids)
    except ValueError as error:
        assert "images_feature length" in str(error)
    else:
        raise AssertionError("Expected a feature-length validation error.")


def test_nonfinite_ratio_and_fractional_grid_are_rejected():
    try:
        FrameVarlenAttentionProjector(8, 8, downsample_ratio=float("inf"), num_attention_heads=2)
    except ValueError as error:
        assert "finite and positive" in str(error)
    else:
        raise AssertionError("Expected a non-finite default ratio error.")

    projector = FrameVarlenAttentionProjector(8, 8, downsample_ratio=4, num_attention_heads=2)
    try:
        projector(
            torch.randn(16, 8),
            torch.tensor([[1.5, 8.0, 8.0]]),
        )
    except ValueError as error:
        assert "positive integers" in str(error)
    else:
        raise AssertionError("Expected a fractional grid dimension error.")

    try:
        projector(
            torch.randn(16, 8),
            torch.tensor([[1, 8, 8]]),
            downsample_ratios=[float("inf")],
        )
    except ValueError as error:
        assert "finite and positive" in str(error)
    else:
        raise AssertionError("Expected a non-finite per-grid ratio error.")


def test_flash_varlen_matches_dense_reference_and_has_finite_gradients():
    _require_cuda()
    torch.manual_seed(7)
    projector = FrameVarlenAttentionProjector(32, 24, 4, 4).cuda().bfloat16()
    qlens, klens = [4, 6, 3], [16, 25, 12]
    q = torch.randn(13, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(53, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = projector._varlen_attention(
        q, k, v, projector._prefix_sums(qlens, q.device),
        projector._prefix_sums(klens, q.device), qlens, klens,
    )
    expected = []
    qo = ko = 0
    for nq, nk in zip(qlens, klens):
        fq = q[qo:qo + nq].float().transpose(0, 1)
        fk = k[ko:ko + nk].float().transpose(0, 1)
        fv = v[ko:ko + nk].float().transpose(0, 1)
        expected.append(((fq @ fk.transpose(-1, -2) / 8**0.5).softmax(-1) @ fv).transpose(0, 1))
        qo += nq
        ko += nk
    expected = torch.cat(expected)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=5e-3)
    actual.float().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def test_cpu_input_and_oversized_heads_are_rejected():
    projector = FrameVarlenAttentionProjector(8, 8, 4, 2)
    try:
        projector(torch.randn(16, 8), torch.tensor([[1, 8, 8]]))
    except (ValueError, RuntimeError) as error:
        assert "CUDA" in str(error) or "flash-attn" in str(error)
    else:
        raise AssertionError("Expected CPU rejection.")
    try:
        FrameVarlenAttentionProjector(4096, 8, 4, 8)
    except ValueError as error:
        assert "head_dim" in str(error)
    else:
        raise AssertionError("Expected oversized attention head rejection.")
