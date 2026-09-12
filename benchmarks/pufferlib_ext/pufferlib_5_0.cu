// Torch binding for PufferLib 5.0's `puff_advantage` CUDA kernel -- float32 only.
//
// PufferLib 5.0 is "0 python" (per Joseph Suarez, PufferLib author, 2026-08-19):
// there is no `torch.ops.pufferlib.*` binding upstream anymore, no
// `pufferlib/extensions/` PyTorch C++ extension, and no PyPI release beyond
// 3.0.0 (PufferLib versions live as git branches -- `refs/heads/4.0`,
// `refs/heads/5.0` -- not PyPI sdists or git tags; see NOTICE.md). The kernel
// itself is called directly from a C/CUDA training loop
// (pufferlib/src/pufferl.cu:1509) with no Python-callable entry point at all.
//
// The __global__ kernel below (`puff_advantage`, lines marked VENDORED) is
// copied VERBATIM from pufferlib/src/algo.cu lines 1524-1602 at commit
// 355e0be1fa7198b6ad7c82df90d0b9d487b4ac59 (branch `5.0`, 2026-08-17) --
// see NOTICE.md for the exact provenance and sha256 of the upstream file.
// Everything else in this file (the torch::Tensor wrapper, shape checks,
// TORCH_LIBRARY registration) is NOT from PufferLib -- it's glue written for
// this benchmark, following the same binding pattern
// benchmarks/pufferlib_ext/pufferlib.cu used for the 3.0.0 kernel, since 5.0
// ships no equivalent binding to copy.
//
// precision_t is `float` unconditionally here (PufferLib's `-DPRECISION_FLOAT`
// build option) -- this repo only benchmarks the float32 regime it actually
// runs in, so the bf16 branch of PufferLib's precision_t/to_float/from_float
// macros (pufferlib/src/pufferl.cu:41-57) is not vendored.
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace pufferlib_5_0 {

// ---- VENDORED (pufferlib/src/algo.cu:1524-1550, float32 precision_t) ----
using precision_t = float;
#define to_float(x) (x)
#define from_float(x) (x)

constexpr int ADV_VEC_WIDTH = 16 / (int)sizeof(precision_t);

__device__ __forceinline__ void adv_ld(const precision_t* p, float* o) {
    float4 v = *(const float4*)p;
    o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
}
__device__ __forceinline__ void adv_st(precision_t* p, const float* o) {
    *(float4*)p = make_float4(o[0], o[1], o[2], o[3]);
}
// ---- end VENDORED block ----

// ---- VENDORED (pufferlib/src/algo.cu:1551-1602) ----
// GAE / full truncated IS (V-trace ρ/c): ρ̄ on δ, c̄ on λ product.
// Same R_{t+1},D_{t+1} indexing as classic puffer_advantage.
__global__ void puff_advantage(const precision_t* values,
        const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages,
        precision_t* returns,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = to_float(values[off + horizon - 1]);
    float next_d = to_float(dones[off + horizon - 1]);
    float next_r = to_float(rewards[off + horizon - 1]);

    for (int seg = horizon / ADV_VEC_WIDTH - 1; seg >= 0; seg--) {
        int base = off + seg * ADV_VEC_WIDTH;
        float v[ADV_VEC_WIDTH], r[ADV_VEC_WIDTH], d[ADV_VEC_WIDTH], imp[ADV_VEC_WIDTH];
        float adv[ADV_VEC_WIDTH] = {};
        float ret[ADV_VEC_WIDTH];
        adv_ld(values + base, v);
        adv_ld(rewards + base, r);
        adv_ld(dones + base, d);
        if (importance) {
            adv_ld(importance + base, imp);
        } else {
            #pragma unroll
            for (int i = 0; i < ADV_VEC_WIDTH; i++) {
                imp[i] = 1.f;
            }
        }
        // Last index H-1 left 0. First seg starts at width-2.
        int i0 = (seg + 1 == horizon / ADV_VEC_WIDTH) ? ADV_VEC_WIDTH - 2 : ADV_VEC_WIDTH - 1;
        #pragma unroll
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.f - next_d;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_r + gamma * next_v * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        #pragma unroll
        for (int i = 0; i < ADV_VEC_WIDTH; i++) {
            ret[i] = v[i] + adv[i];
        }
        adv_st(advantages + base, adv);
        adv_st(returns + base, ret);
    }
}
// ---- end VENDORED block ----

#undef to_float
#undef from_float

// ---- Benchmark glue (not from PufferLib) ----

void puff_advantage_5_0_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        torch::Tensor returns, int num_steps, int horizon) {
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages, returns}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        TORCH_CHECK(t.is_contiguous(), "All tensors must be contiguous");
    }
    TORCH_CHECK(horizon % ADV_VEC_WIDTH == 0,
        "puff_advantage 5.0's vectorized load/store requires horizon to be a "
        "multiple of ", ADV_VEC_WIDTH, " (16B segments over float32); got horizon=", horizon);
}

void compute_puff_advantage_5_0_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        torch::Tensor returns, double gamma, double lambda, double rho_clip,
        double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    puff_advantage_5_0_check(values, rewards, dones, importance, advantages,
        returns, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    // ADV_THREADS=64, matching pufferlib/src/pufferl.cu:1485's real launch config.
    constexpr int ADV_THREADS = 64;
    int blocks = (num_steps + ADV_THREADS - 1) / ADV_THREADS;

    puff_advantage<<<blocks, ADV_THREADS>>>(
        values.data_ptr<float>(),
        rewards.data_ptr<float>(),
        dones.data_ptr<float>(),
        importance.data_ptr<float>(),
        advantages.data_ptr<float>(),
        returns.data_ptr<float>(),
        (float)gamma,
        (float)lambda,
        (float)rho_clip,
        (float)c_clip,
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

TORCH_LIBRARY(pufferlib_5_0, m) {
   m.def("compute_puff_advantage_5_0(Tensor(a!) values, Tensor(b!) rewards, "
         "Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, "
         "Tensor(f!) returns, float gamma, float lambda, float rho_clip, "
         "float c_clip) -> ()");
}

TORCH_LIBRARY_IMPL(pufferlib_5_0, CUDA, m) {
  m.impl("compute_puff_advantage_5_0", &compute_puff_advantage_5_0_cuda);
}

}  // namespace pufferlib_5_0
