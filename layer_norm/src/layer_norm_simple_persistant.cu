#include "utils.h"

template<size_t BLOCK_SIZE>
__global__ void _persistant_layer_norm(float* __restrict__ Y,
                                         float* __restrict__ X,
                                         float* __restrict__ gamma,
                                         float* __restrict__ beta,
                                         size_t B, size_t F, size_t D1, size_t D2,
                                         size_t stride_b, float* __restrict__ batch_mean, float* __restrict__ batch_std)
{
    __shared__ float mean_smem[BLOCK_SIZE];
    __shared__ float std_smem[BLOCK_SIZE];
    constexpr int VECTOR_LOAD = 4;
    const int batch_idx = (blockIdx.x % B);
    const int cluster_idx = (blockIdx.x / B);
    const int extra = (blockIdx.x < (gridDim.x % B)) ? 1 : 0;
    const int cluster_dim = gridDim.x / B + extra;
    const int block_size_eff = cluster_dim * BLOCK_SIZE;
    const int vector_size = stride_b / VECTOR_LOAD;

    float* X_base = X + (batch_idx * stride_b);
    float* Y_base = Y + (batch_idx * stride_b);
    
    float mean(0.0f), std(0.0f);
    float4 x;
    for(int idx = 0; idx < vector_size; idx += block_size_eff){
        int global_idx = idx + threadIdx.x;
        if (global_idx >= vector_size) continue;
        x = reinterpret_cast<float4*>(X_base)[global_idx];
        mean += x.x;
        mean += x.y;
        mean += x.z;
        mean += x.w;
        std  += (x.x * x.x);
        std  += (x.y * x.y);
        std  += (x.z * x.z);
        std  += (x.w * x.w);
    }
    mean_smem[threadIdx.x] = mean;
    std_smem[threadIdx.x] = std;
    __syncthreads();
    block_reduce<float, BLOCK_SIZE>(mean_smem);
    block_reduce<float, BLOCK_SIZE>(std_smem);
    mean = mean_smem[0];
    std  = std_smem[0];
    atomicAdd(&batch_mean[batch_idx], mean);
    atomicAdd(&batch_std[batch_idx], std);
    __syncthreads();
    mean = batch_mean[batch_idx];
    std  = batch_std[batch_idx];
    mean = mean / (float)stride_b;
    std = std / (float)stride_b;
    std = rsqrtf(std - (mean * mean) + 1e-5f);

    float4 g, b, y ;
    for(int idx = 0; idx < vector_size; idx += block_size_eff){
        int global_idx = idx + threadIdx.x;
        if (global_idx >= vector_size) continue;
        x = reinterpret_cast<float4*>(X_base)[global_idx];
        g = reinterpret_cast<float4*>(gamma)[global_idx];
        b = reinterpret_cast<float4*>(beta)[global_idx];
        y.x = ((x.x - mean) * std) * g.x + b.x;
        y.y = ((x.y - mean) * std) * g.y + b.y;
        y.z = ((x.z - mean) * std) * g.z + b.z;
        y.w = ((x.w - mean) * std) * g.w + b.w;
        reinterpret_cast<float4*>(Y_base)[global_idx] = y;
    }
}


void layer_norm_simple_persistant(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    const size_t STRIDE = F * D1 * D2;
    constexpr size_t num_sms = 70;
    cudaStream_t s;
    cudaStreamCreate(&s);
    float* batch_mean;
    cudaMallocAsync(&batch_mean, 2*B*sizeof(float), s);
    float* batch_std=batch_mean+B;
    cudaMemsetAsync(batch_mean, 0, 2*B*sizeof(float), s);

    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t MIN_BLOCK_SIZE = 32;
    const size_t LENGTH = std::max<size_t>(MIN_BLOCK_SIZE, next_power_of_2(STRIDE));
    const size_t BLOCK_SIZE = std::min<size_t>(MAX_BLOCK_SIZE, LENGTH);
    dim3 grid(num_sms);
    dim3 block(BLOCK_SIZE);
    switch (BLOCK_SIZE) {
        case 32:
            _persistant_layer_norm<32><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);
        case 64:
            _persistant_layer_norm<64><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);
        case 128:
            _persistant_layer_norm<128><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);
        case 256:
            _persistant_layer_norm<256><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);
        case 512:
            _persistant_layer_norm<512><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);
        case 1024:
            _persistant_layer_norm<1024><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE, batch_mean, batch_std);    
    }
    cudaDeviceSynchronize();
}