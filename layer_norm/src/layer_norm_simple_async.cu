#include <cuda/pipeline>
#include <cuda/barrier>
#include "utils.h"

template<int BLOCK_SIZE, int FACTOR>
__global__ void _layer_norm_simple_async(float* __restrict__ Y,
                                         const float* __restrict__ X,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         size_t B, size_t F, size_t D1, size_t D2,
                                         size_t stride_b)
{
    extern __shared__ float sram[];
    float* smem       = sram;
    float* gamma_smem = sram       + (2 * FACTOR * BLOCK_SIZE);
    float* beta_smem  = gamma_smem + (2 * FACTOR * BLOCK_SIZE);
    float* mean = gamma_smem;
    float* var  = beta_smem;
    int num_vectors = stride_b >> 2; 
    constexpr int SMEM_STRIDE = (FACTOR * BLOCK_SIZE);
    const int LOOPS = int((stride_b + SMEM_STRIDE - 1) / SMEM_STRIDE); 
    const float* X_block = X + blockIdx.x * stride_b;
    float*       Y_block = Y + blockIdx.x * stride_b;

    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];
    cuda::barrier<cuda::thread_scope_block>::arrival_token token[2];

    auto aligned_mem = cuda::aligned_size_t<16>(SMEM_STRIDE * sizeof(float));
    
    if (threadIdx.x == 0) {
        init(&bar[0], 1);
        init(&bar[1], 1);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, aligned_mem);
        cuda::device::memcpy_async_tx(smem, X_block, aligned_mem, bar[0]);
    }
    float4 x;
    float mean_acc, var_acc;
    int stage, next_stage;
    for (int i = 0; i < LOOPS - 1; ++i) {
        stage      = (i & 1);
        next_stage = (i + 1) & 1;
        if (threadIdx.x == 0) {
            token[next_stage] = cuda::device::barrier_arrive_tx(bar[next_stage], 1, aligned_mem);
            cuda::device::memcpy_async_tx(smem + next_stage * SMEM_STRIDE,
                                          X_block + (i + 1) * SMEM_STRIDE,
                                          aligned_mem,
                                          bar[next_stage]);
            bar[stage].wait(cuda::std::move(token[stage]));
        } 
        __syncthreads();
        int global_idx = (i * BLOCK_SIZE) + threadIdx.x;
        if (global_idx >= num_vectors) continue;
        x = reinterpret_cast<float4*>(&smem[stage * SMEM_STRIDE])[threadIdx.x];
        mean_acc += x.x;
        mean_acc += x.y;
        mean_acc += x.z;
        mean_acc += x.w;
        var_acc  += x.x * x.x;
        var_acc  += x.y * x.y;
        var_acc  += x.z * x.z;
        var_acc  += x.w * x.w;
    }
    mean[threadIdx.x] = mean_acc;
    var[threadIdx.x] = var_acc;
    __syncthreads();
    block_reduce<float, BLOCK_SIZE>(mean);
    block_reduce<float, BLOCK_SIZE>(var);

    const float Nf   = float(stride_b);
    float mean_f     = mean[0] / Nf;
    float ex2        = var[0]  / Nf;
    float inv_std    = rsqrtf(ex2 - mean_f * mean_f + 1e-5f);

    auto aligned_mem_3  = cuda::aligned_size_t<16>(3 * SMEM_STRIDE * sizeof(float));
    if (threadIdx.x == 0) { 
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, aligned_mem_3);
        cuda::device::memcpy_async_tx(smem,       X_block, aligned_mem, bar[0]);
        cuda::device::memcpy_async_tx(gamma_smem, gamma,   aligned_mem, bar[0]);
        cuda::device::memcpy_async_tx(beta_smem,  beta,    aligned_mem, bar[0]);
    }

    float4 y, g, b;
    int i = 0;
    #pragma unroll
    for (i = 0; i < LOOPS; ++i) {
        stage      = (i & 1);
        next_stage = (i + 1) & 1;
        if (threadIdx.x == 0 && i < LOOPS - 1) {
            token[next_stage] = cuda::device::barrier_arrive_tx(bar[next_stage], 1, aligned_mem_3);
            cuda::device::memcpy_async_tx(smem       + next_stage * SMEM_STRIDE, X_block + (i + 1) * SMEM_STRIDE, aligned_mem, bar[next_stage]);
            cuda::device::memcpy_async_tx(gamma_smem + next_stage * SMEM_STRIDE, gamma   + (i + 1) * SMEM_STRIDE, aligned_mem, bar[next_stage]);
            cuda::device::memcpy_async_tx(beta_smem  + next_stage * SMEM_STRIDE, beta    + (i + 1) * SMEM_STRIDE, aligned_mem, bar[next_stage]);
            bar[stage].wait(cuda::std::move(token[stage]));
        } 
        __syncthreads();
        int global_idx = (i * BLOCK_SIZE) + threadIdx.x;
        if (global_idx >= num_vectors) continue;
        x = reinterpret_cast<float4*>(&smem[stage * SMEM_STRIDE])[threadIdx.x];
        g = reinterpret_cast<float4*>(&gamma_smem[stage * SMEM_STRIDE])[threadIdx.x];
        b = reinterpret_cast<float4*>(&beta_smem[stage * SMEM_STRIDE])[threadIdx.x];
        y.x = (x.x - mean_f) * inv_std * g.x + b.x;
        y.y = (x.y - mean_f) * inv_std * g.y + b.y;
        y.z = (x.z - mean_f) * inv_std * g.z + b.z;
        y.w = (x.w - mean_f) * inv_std * g.w + b.w;
        reinterpret_cast<float4*>(Y_block + i * SMEM_STRIDE)[threadIdx.x] = y;
    }
}

void layer_norm_simple_async(float *X, float *Y,
                             float *gamma, float *beta,
                             size_t B, size_t F, size_t D1, size_t D2)
{
    constexpr int FACTOR        = 4;
    constexpr size_t MAX_BLOCK_SIZE = 1024;
    
    const size_t STRIDE         = F * D1 * D2;
    size_t LENGTH               = (STRIDE + FACTOR - 1) / FACTOR;
    LENGTH                      = std::max<size_t>(32, next_power_of_2(LENGTH));
    const size_t BLOCK_SIZE     = std::min<size_t>(MAX_BLOCK_SIZE, LENGTH);
    const size_t SMEM_FLOATS    = 2 * FACTOR * BLOCK_SIZE * 3;
    const size_t SMEM_BYTES     = SMEM_FLOATS * sizeof(float);

    dim3 grid(B), block(BLOCK_SIZE);

    switch (BLOCK_SIZE) {
        case 32:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<32,  FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<32,  FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
        case 64:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<64,  FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<64,  FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
        case 128:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<128, FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<128, FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
        case 256:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<256, FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<256, FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
        case 512:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<512, FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<512, FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
        case 1024:
            cudaFuncSetAttribute((const void*)_layer_norm_simple_async<1024, FACTOR>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)SMEM_BYTES);
            _layer_norm_simple_async<1024, FACTOR><<<grid, block, SMEM_BYTES>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
            break;
    }
}
