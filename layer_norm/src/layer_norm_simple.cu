#include "utils.h"

template<int BLOCK_SIZE>
__global__ void _layer_norm_simple(float* Y, float* X, float* gamma, float* beta, 
                            size_t B, size_t F, size_t D1, size_t D2, 
                            size_t stride_b)
{
    constexpr int LOAD_SIZE = 4;
    
    __shared__ float mean_smem[BLOCK_SIZE]; 
    __shared__ float std_smem[BLOCK_SIZE]; 

    float4* batch     = reinterpret_cast<float4*>(&X[blockIdx.x * stride_b]); 
    float4* out_batch = reinterpret_cast<float4*>(&Y[blockIdx.x * stride_b]); 
    
    float scale_factor = (1.0f / stride_b); 
    int num_vectors = stride_b / LOAD_SIZE;
    
    float mean(0.0f), std(0.0f); 
    
    for(int idx = 0 ; idx < num_vectors; idx += BLOCK_SIZE){
        int global_idx = idx + threadIdx.x;
        if (global_idx >= num_vectors) continue;
        float4 ele = batch[global_idx]; 
        mean += ele.x; 
        mean += ele.y;
        mean += ele.z;
        mean += ele.w;
        std  += ele.x * ele.x;
        std  += ele.y * ele.y;
        std  += ele.z * ele.z;
        std  += ele.w * ele.w;
    }

    std_smem[threadIdx.x]   = std; 
    mean_smem[threadIdx.x]  = mean;
    __syncthreads(); 
    block_reduce<float, BLOCK_SIZE>(std_smem);
    block_reduce<float, BLOCK_SIZE>(mean_smem);
    __syncthreads();
    mean = mean_smem[0] * scale_factor; 
    std  = std_smem[0]  * scale_factor;
    std  = rsqrtf(std - (mean * mean) + 1e-5f);
    
    for(int idx = 0; idx < num_vectors; idx += BLOCK_SIZE){
        int global_idx = idx + threadIdx.x;
        if (global_idx >= num_vectors) continue;
        float4 ele = reinterpret_cast<float4*>(batch)[global_idx]; 
        float4 g   = reinterpret_cast<float4*>(gamma)[global_idx]; 
        float4 b   = reinterpret_cast<float4*>(beta)[global_idx]; 
        float4 out; 
        out.x = ((ele.x - mean) * std) * g.x + b.x; 
        out.y = ((ele.y - mean) * std) * g.y + b.y; 
        out.z = ((ele.z - mean) * std) * g.z + b.z; 
        out.w = ((ele.w - mean) * std) * g.w + b.w; 
        out_batch[global_idx] = out; 
    }
}


void layer_norm_simple(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    const size_t STRIDE = F * D1 * D2;
    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t MIN_BLOCK_SIZE = 32;
    const size_t LENGTH = std::max<size_t>(MIN_BLOCK_SIZE, next_power_of_2(STRIDE));
    const size_t BLOCK_SIZE = std::min<size_t>(MAX_BLOCK_SIZE, LENGTH);
    dim3 grid(B);
    dim3 block(BLOCK_SIZE);
    switch (BLOCK_SIZE) {
        case 32:
            _layer_norm_simple<32><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 64:
            _layer_norm_simple<64><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 128:
            _layer_norm_simple<128><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 256:
            _layer_norm_simple<256><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 512:
            _layer_norm_simple<512><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 1024:
            _layer_norm_simple<1024><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);    
    }
    cudaDeviceSynchronize();

}
