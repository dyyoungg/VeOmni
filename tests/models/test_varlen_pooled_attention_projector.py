from unittest import SkipTest

import pytest
import torch
import torch.nn.functional as F

from veomni.models.custom.llava_qwen3moe.projector_pooled_query_attention import (
    PooledQueryAttentionProjector,
    PoolingResidualAttentionProjector,
    broadcast_pooling_windows,
    flash_attn_varlen_func,
    get_adaptive_pool_size,
)


PROJECTOR_CLASSES = [PooledQueryAttentionProjector, PoolingResidualAttentionProjector]


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
        pooled = F.adaptive_avg_pool2d(grid_features.permute(0, 3, 1, 2), (output_height, output_width))
        outputs.append(pooled.permute(0, 2, 3, 1).reshape(-1, features.shape[-1]))
        seq_len.extend([output_height * output_width] * time)
        offset += length
    if outputs:
        return torch.cat(outputs), seq_len
    return features.new_empty((0, features.shape[-1])), seq_len


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_zero_initialized_attention_matches_dynamic_average_pooling(projector_class):
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
    projector = projector_class(16, 16, downsample_ratio=4, num_attention_heads=4)

    projector = projector.cuda().bfloat16()
    features = features.cuda().bfloat16()
    enhanced, seq_len, aux = projector.enhance_tokens(
        features,
        grids,
        merge_size=merge_sizes,
        downsample_ratios=ratios,
    )
    expected, expected_seq_len = _adaptive_avgpool_reference(features, grids, merge_sizes, ratios)

    torch.testing.assert_close(enhanced, expected)
    torch.testing.assert_close(aux.base_tokens, expected)
    assert torch.count_nonzero(aux.residual_tokens) == 0
    assert seq_len == expected_seq_len


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_frame_varlen_boundaries_follow_packed_grids(projector_class):
    _require_cuda()
    grids = torch.tensor([[2, 8, 12], [1, 12, 8]])
    merge_sizes = [2, 2]
    ratios = [4.0, 3.0]
    feature_count = sum(
        time * (height // merge_size) * (width // merge_size)
        for (time, height, width), merge_size in zip(grids.tolist(), merge_sizes)
    )
    projector = projector_class(8, 12, downsample_ratio=4, num_attention_heads=2)

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
    expected_query_lengths = [first_output[0] * first_output[1]] * 2 + [second_output[0] * second_output[1]]
    expected_key_value_lengths = [4 * 6, 4 * 6, 6 * 4]

    assert seq_len == expected_query_lengths
    assert aux.query_lengths == expected_query_lengths
    assert aux.key_value_lengths == expected_key_value_lengths
    assert aux.cu_seqlens_q.tolist() == [0, 6, 12, 18]
    assert aux.cu_seqlens_kv.tolist() == [0, 24, 48, 72]
    assert output.shape == (sum(expected_query_lengths), 12)


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_frames_do_not_exchange_information(projector_class):
    _require_cuda()
    torch.manual_seed(1)
    grids = torch.tensor([[2, 8, 8]])
    features = torch.randn(32, 8)
    projector = projector_class(8, 8, downsample_ratio=4, num_attention_heads=2)
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
    torch.testing.assert_close(first_output[:tokens_per_frame], second_output[:tokens_per_frame])
    assert not torch.allclose(first_output[tokens_per_frame:], second_output[tokens_per_frame:])


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_attention_residual_and_qkv_receive_gradients(projector_class):
    _require_cuda()
    torch.manual_seed(2)
    grids = torch.tensor([[1, 8, 8]])
    features = torch.randn(16, 8, requires_grad=True)
    projector = projector_class(8, 6, downsample_ratio=4, num_attention_heads=2)
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


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_meta_materialization_restores_zero_residual_projection(projector_class):
    with torch.device("meta"):
        projector = projector_class(8, 8, downsample_ratio=4, num_attention_heads=2)

    assert projector.attention_out.weight.is_meta
    projector.to_empty(device="cpu")

    assert torch.count_nonzero(projector.attention_out.weight) == 0
    assert projector.attention_out.weight._veomni_fsdp_shard_dim == 1


@pytest.mark.parametrize("projector_class", PROJECTOR_CLASSES)
def test_empty_packed_visual_input_is_supported(projector_class):
    _require_cuda()
    projector = projector_class(8, 6, downsample_ratio=4, num_attention_heads=2)

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
    projector = PooledQueryAttentionProjector(8, 8, downsample_ratio=4, num_attention_heads=2)
    grids = torch.tensor([[1, 8, 8]])

    try:
        projector(torch.randn(15, 8), grids)
    except ValueError as error:
        assert "images_feature length" in str(error)
    else:
        raise AssertionError("Expected a feature-length validation error.")


def test_nonfinite_ratio_and_fractional_grid_are_rejected():
    try:
        PooledQueryAttentionProjector(8, 8, downsample_ratio=float("inf"), num_attention_heads=2)
    except ValueError as error:
        assert "finite and positive" in str(error)
    else:
        raise AssertionError("Expected a non-finite default ratio error.")

    projector = PooledQueryAttentionProjector(8, 8, downsample_ratio=4, num_attention_heads=2)
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
    projector = PooledQueryAttentionProjector(32, 24, 4, 4).cuda().bfloat16()
    qlens, klens = [4, 6, 3], [16, 25, 12]
    q = torch.randn(13, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(53, 4, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    actual = projector._varlen_attention(
        q,
        k,
        v,
        projector._prefix_sums(qlens, q.device),
        projector._prefix_sums(klens, q.device),
        qlens,
        klens,
    )
    expected = []
    qo = ko = 0
    for nq, nk in zip(qlens, klens):
        fq = q[qo : qo + nq].float().transpose(0, 1)
        fk = k[ko : ko + nk].float().transpose(0, 1)
        fv = v[ko : ko + nk].float().transpose(0, 1)
        expected.append(((fq @ fk.transpose(-1, -2) / 8**0.5).softmax(-1) @ fv).transpose(0, 1))
        qo += nq
        ko += nk
    expected = torch.cat(expected)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=5e-3)
    actual.float().square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def test_cpu_input_and_oversized_heads_are_rejected():
    projector = PooledQueryAttentionProjector(8, 8, 4, 2)
    try:
        projector(torch.randn(16, 8), torch.tensor([[1, 8, 8]]))
    except (ValueError, RuntimeError) as error:
        assert "CUDA" in str(error) or "flash-attn" in str(error)
    else:
        raise AssertionError("Expected CPU rejection.")
    try:
        PooledQueryAttentionProjector(4096, 8, 4, 8)
    except ValueError as error:
        assert "head_dim" in str(error)
    else:
        raise AssertionError("Expected oversized attention head rejection.")


@pytest.mark.parametrize("height,width,ho,wo", [(4, 6, 2, 3), (5, 7, 2, 3), (3, 4, 1, 1), (3, 5, 3, 5)])
def test_broadcast_matches_actual_pooling_window_coverage(height, width, ho, wo):
    pooled = torch.randn(2, ho, wo, 3, dtype=torch.float64, requires_grad=True)
    expected = torch.zeros(2, height, width, 3, dtype=torch.float64)
    counts = torch.zeros(height, width, dtype=torch.float64)
    for i in range(ho):
        y0, y1 = i * height // ho, ((i + 1) * height + ho - 1) // ho
        for j in range(wo):
            x0, x1 = j * width // wo, ((j + 1) * width + wo - 1) // wo
            expected[:, y0:y1, x0:x1] += pooled[:, i : i + 1, j : j + 1]
            counts[y0:y1, x0:x1] += 1
    expected = expected / counts[None, :, :, None]
    actual = broadcast_pooling_windows(pooled, height, width)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.gradcheck(lambda x: broadcast_pooling_windows(x, height, width), (pooled,))


def test_overlap_broadcast_averages_neighboring_coarse_tokens():
    pooled = torch.tensor([[[[10.0]], [[20.0]]]])
    actual = broadcast_pooling_windows(pooled, 5, 1)
    torch.testing.assert_close(actual.flatten(), torch.tensor([10.0, 10.0, 15.0, 20.0, 20.0]))


def test_pooling_residuals_reconstruct_input_and_have_zero_cell_mean():
    projector = PoolingResidualAttentionProjector(8, 6, 4, 2)
    features = torch.randn(2, 4, 6, 8, requires_grad=True)
    base, residual = projector._pool_grid(features, 2, 3)
    torch.testing.assert_close(features, broadcast_pooling_windows(base, 4, 6) + residual)
    cell_mean = F.adaptive_avg_pool2d(residual.permute(0, 3, 1, 2), (2, 3))
    torch.testing.assert_close(cell_mean, torch.zeros_like(cell_mean), atol=2e-7, rtol=0)
    # No per-token normalization: value magnitude must scale with residual magnitude.
    projected = projector._project_values(residual.reshape(-1, 8), features.reshape(-1, 8))
    doubled = projector._project_values(2 * residual.reshape(-1, 8), features.reshape(-1, 8))
    torch.testing.assert_close(doubled, 2 * projected)
    projected.square().sum().backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


@pytest.mark.parametrize("height,width,ho,wo", [(4, 6, 2, 3), (5, 7, 2, 3), (3, 5, 3, 5)])
def test_constant_spatial_features_have_no_pooling_residual(height, width, ho, wo):
    projector = PoolingResidualAttentionProjector(8, 6, 4, 2)
    features = torch.randn(2, 1, 1, 8).expand(-1, height, width, -1).contiguous()
    _, residual = projector._pool_grid(features, ho, wo)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=5e-7, rtol=0)


def test_packed_residual_values_use_each_grids_pooling_baseline():
    projector = PoolingResidualAttentionProjector(8, 6, 4, 2)
    grids = torch.tensor([[2, 8, 12], [1, 10, 14]])
    features = torch.randn(2 * 4 * 6 + 5 * 7, 8)
    metadata = projector._build_frame_metadata(features, grids, [2, 2], [4.0, 4.0])
    expected = []
    offset = 0
    for time, height, width in [(2, 4, 6), (1, 5, 7)]:
        length = time * height * width
        grid_features = features[offset : offset + length].view(time, height, width, 8)
        ho, wo = get_adaptive_pool_size(height, width, 4.0)
        _, residual = projector._pool_grid(grid_features, ho, wo)
        expected.append(residual.reshape(-1, 8))
        offset += length
    torch.testing.assert_close(metadata[-1], torch.cat(expected))


def test_flash_attention_reads_raw_pooling_residual_values():
    _require_cuda()
    projector = PoolingResidualAttentionProjector(8, 6, 4, 2).cuda().bfloat16()
    features = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    grids = torch.tensor([[1, 8, 8]])
    with torch.no_grad():
        projector.attention_out.weight.copy_(torch.eye(8))
        actual, _, aux = projector.enhance_tokens(features, grids)
        metadata = projector._build_frame_metadata(features, grids, [2], [4.0])
        base, qpos, kpos, rates = metadata[:4]
        q = projector.query_proj(projector.query_norm(base))
        q = q + projector.query_position_proj(qpos.to(features.dtype))
        q = q + projector.rate_proj(rates.to(features.dtype))
        k = projector.key_proj(projector.key_value_norm(features))
        k = k + projector.key_position_proj(kpos.to(features.dtype))
        v = projector.value_proj(aux.dense_residual_tokens)
        q, k, v = [x.view(-1, 2, 4).float().transpose(0, 1) for x in (q, k, v)]
        attended = ((q @ k.transpose(-1, -2) / 4**0.5).softmax(-1) @ v).transpose(0, 1).reshape(-1, 8)
        torch.testing.assert_close(aux.residual_tokens.float(), attended, atol=5e-3, rtol=2e-2)
        torch.testing.assert_close(actual, base + aux.residual_tokens)
