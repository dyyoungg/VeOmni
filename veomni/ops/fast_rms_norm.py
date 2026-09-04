"""FastRMSNorm v2: self-contained HIP fwd+bwd (no liger internals at all).

fwd: 320 threads x 16 elems (2x uint4), 92% measured HBM bandwidth.
bwd: persistent grid (sm_count blocks), per-thread 16-col dW fp32 register
     accumulator, in-row shuffle reduce for t, dW via [nblk, N] partial + sum.
Math:
  t[i]        = sum_j dY[i,j]*W[j]*X[i,j]
  dX[i,j]     = RSTD[i]*dY[i,j]*W[j] - (RSTD[i]^3/N)*X[i,j]*t[i]
  dW[j]       = sum_i dY[i,j]*X[i,j]*RSTD[i]
Env-gated: VEOMNI_FAST_RMS_NORM=1 (integration in qwen2 gpu_patch).
"""
import os

import torch
import torch.nn as nn

os.environ.setdefault("CUDA_HOME", "/opt/dtk/cuda/cuda")
os.environ["PATH"] = "/opt/dtk/cuda/cuda/bin:/opt/dtk/bin:/opt/dtk-26.04/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
os.environ.setdefault("MAX_JOBS", "2")

from torch.utils.cpp_extension import load_inline  # noqa: E402

_CPP_SRC = (
    "void rms_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor rstd, double eps);\n"
    "void rms_bwd(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor RSTD,"
    " torch::Tensor dX, torch::Tensor dWp);\n"
)

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

#define THREADS 320
#define VEC 16  // bf16 elems per thread; N=5120: 320*16 exact fit

__global__ void __launch_bounds__(THREADS)
rms_fwd_kernel(const __nv_bfloat16* __restrict__ X,
               const __nv_bfloat16* __restrict__ W,
               __nv_bfloat16* __restrict__ Y,
               float* __restrict__ RSTD,
               const int N, const float eps) {
    const long row = blockIdx.x;
    const __nv_bfloat16* xr = X + row * (long)N;
    __nv_bfloat16* yr = Y + row * (long)N;
    const int tid = threadIdx.x;

    __nv_bfloat16 xv[VEC];
    {
        uint4* d = reinterpret_cast<uint4*>(xv);
        const uint4* s = reinterpret_cast<const uint4*>(xr + tid * VEC);
        d[0] = s[0];
        d[1] = s[1];
    }
    float partial = 0.f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) {
        float f = __bfloat162float(xv[i]);
        partial += f * f;
    }
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);

    __shared__ float s_red[THREADS / 64];
    __shared__ float s_inv;
    const int wid = tid >> 6;
    if ((tid & 63) == 0) s_red[wid] = partial;
    __syncthreads();
    if (tid == 0) {
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < THREADS / 64; ++i) s += s_red[i];
        s_inv = rsqrtf(s / (float)N + eps);
        RSTD[row] = s_inv;
    }
    __syncthreads();
    const float inv = s_inv;

    __nv_bfloat16 wv[VEC];
    {
        uint4* d = reinterpret_cast<uint4*>(wv);
        const uint4* s = reinterpret_cast<const uint4*>(W + tid * VEC);
        d[0] = s[0];
        d[1] = s[1];
    }
    __nv_bfloat16 yv[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i)
        yv[i] = __float2bfloat16(__bfloat162float(xv[i]) * inv * __bfloat162float(wv[i]));
    {
        uint4* d = reinterpret_cast<uint4*>(yr + tid * VEC);
        const uint4* s = reinterpret_cast<const uint4*>(yv);
        d[0] = s[0];
        d[1] = s[1];
    }
}

__global__ void __launch_bounds__(THREADS)
rms_bwd_kernel(const __nv_bfloat16* __restrict__ dY,
               const __nv_bfloat16* __restrict__ X,
               const __nv_bfloat16* __restrict__ W,
               const float* __restrict__ RSTD,
               __nv_bfloat16* __restrict__ dX,
               float* __restrict__ dWp,  // [gridDim.x, N]
               const long M, const int N) {
    const int tid = threadIdx.x;
    const int nblk = gridDim.x;
    const long rpp = (M + nblk - 1) / nblk;
    const long row0 = (long)blockIdx.x * rpp;
    const long row1 = min(row0 + rpp, M);

    __nv_bfloat16 wv[VEC];
    {
        uint4* d = reinterpret_cast<uint4*>(wv);
        const uint4* s = reinterpret_cast<const uint4*>(W + tid * VEC);
        d[0] = s[0];
        d[1] = s[1];
    }
    float wf[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) wf[i] = __bfloat162float(wv[i]);

    float dw_acc[VEC];
    #pragma unroll
    for (int i = 0; i < VEC; ++i) dw_acc[i] = 0.f;

    __shared__ float s_red[THREADS / 64];
    __shared__ float s_t;

    for (long row = row0; row < row1; ++row) {
        const __nv_bfloat16* dyr = dY + row * (long)N;
        const __nv_bfloat16* xr = X + row * (long)N;
        __nv_bfloat16* dxr = dX + row * (long)N;
        const float rstd = RSTD[row];

        __nv_bfloat16 dyv[VEC], xv[VEC];
        {
            uint4* d = reinterpret_cast<uint4*>(dyv);
            const uint4* s = reinterpret_cast<const uint4*>(dyr + tid * VEC);
            d[0] = s[0];
            d[1] = s[1];
        }
        {
            uint4* d = reinterpret_cast<uint4*>(xv);
            const uint4* s = reinterpret_cast<const uint4*>(xr + tid * VEC);
            d[0] = s[0];
            d[1] = s[1];
        }
        float partial = 0.f;
        float dyf[VEC], xf[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; ++i) {
            dyf[i] = __bfloat162float(dyv[i]);
            xf[i] = __bfloat162float(xv[i]);
            partial += dyf[i] * wf[i] * xf[i];
            dw_acc[i] += dyf[i] * (xf[i] * rstd);
        }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffffffffffULL, partial, off);
        if ((tid & 63) == 0) s_red[tid >> 6] = partial;
        __syncthreads();
        if (tid == 0) {
            float t = 0.f;
            #pragma unroll
            for (int i = 0; i < THREADS / 64; ++i) t += s_red[i];
            s_t = t;
        }
        __syncthreads();
        const float t = s_t;

        const float c = rstd * rstd * rstd * t / (float)N;
        __nv_bfloat16 dxv[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; ++i)
            dxv[i] = __float2bfloat16(rstd * dyf[i] * wf[i] - c * xf[i]);
        {
            uint4* d = reinterpret_cast<uint4*>(dxv);
            uint4* o = reinterpret_cast<uint4*>(dxr + tid * VEC);
            o[0] = d[0];
            o[1] = d[1];
        }
    }

    float* dwrow = dWp + (long)blockIdx.x * N + tid * VEC;
    float4* dst = reinterpret_cast<float4*>(dwrow);
    dst[0] = make_float4(dw_acc[0], dw_acc[1], dw_acc[2], dw_acc[3]);
    dst[1] = make_float4(dw_acc[4], dw_acc[5], dw_acc[6], dw_acc[7]);
    dst[2] = make_float4(dw_acc[8], dw_acc[9], dw_acc[10], dw_acc[11]);
    dst[3] = make_float4(dw_acc[12], dw_acc[13], dw_acc[14], dw_acc[15]);
}

void rms_fwd(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor rstd, double eps) {
    const int N = x.size(-1);
    const long M = x.numel() / N;
    TORCH_CHECK(N == 5120, "fast rms fwd specialized for N=5120, got ", N);
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_fwd_kernel<<<M, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        rstd.data_ptr<float>(),
        N, (float)eps);
}

void rms_bwd(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor RSTD,
             torch::Tensor dX, torch::Tensor dWp) {
    const int N = X.size(-1);
    const long M = X.numel() / N;
    TORCH_CHECK(N == 5120, "fast rms bwd specialized for N=5120, got ", N);
    const int nblk = X.get_device() >= 0 ?
        (int)at::cuda::getCurrentDeviceProperties()->multiProcessorCount : 80;
    TORCH_CHECK(M > 0, "empty input");
    auto stream = at::cuda::getCurrentCUDAStream();
    rms_bwd_kernel<<<nblk, THREADS, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dY.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(W.data_ptr()),
        RSTD.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dX.data_ptr()),
        dWp.data_ptr<float>(),
        M, N);
}
"""

_ext = load_inline(
    name="fast_rms_v2",
    cpp_sources=_CPP_SRC,
    cuda_sources=_CUDA_SRC,
    functions=["rms_fwd", "rms_bwd"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

_SM_COUNT = None


def _sm_count():
    global _SM_COUNT
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
    return _SM_COUNT


class _FastRMSNormFunction(torch.autograd.Function):
    """fully self-contained: custom HIP forward + custom HIP backward."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        x = x.contiguous()
        out = torch.empty_like(x)
        n_rows = x.numel() // x.shape[-1]
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        _ext.rms_fwd(x, weight, out, rstd, eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dY):
        X, W, RSTD = ctx.saved_tensors
        dY = dY.contiguous()
        n_rows = X.numel() // X.shape[-1]
        dX = torch.empty_like(X)
        nblk = min(_sm_count(), n_rows)
        dWp = torch.empty(nblk, X.shape[-1], dtype=torch.float32, device=X.device)
        _ext.rms_bwd(dY, X, W, RSTD, dX, dWp)
        dW = dWp.sum(0).to(W.dtype)
        return dX.view_as(X), dW, None


class FastRMSNorm(nn.Module):
    """Drop-in replacement for LigerRMSNorm (hidden=5120)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        if hidden_size != 5120:
            raise ValueError(f"FastRMSNorm only supports hidden=5120, got {hidden_size}")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        return _FastRMSNormFunction.apply(x, self.weight, self.variance_epsilon)
