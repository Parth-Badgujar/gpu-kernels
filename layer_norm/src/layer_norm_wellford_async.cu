#include <cuda/pipeline>
#include <cuda/barrier>
#include "utils.h"

template<int BLOCK_SIZE>
__global__ void _layer_norm_wellford(float* Y, float* X, float* gamma, float* beta, 
                            size_t B, size_t F, size_t D1, size_t D2, 
                            const int stride_b)
{ 
    alignas(16) __shared__ float smem[2][BLOCK_SIZE];
    alignas(16) __shared__ float gamma_smem[2][BLOCK_SIZE];
    alignas(16) __shared__ float beta_smem [2][BLOCK_SIZE];

    __shared__ float mean[BLOCK_SIZE];
    __shared__ float m2  [BLOCK_SIZE];
    
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    const int bts = sizeof(float) * BLOCK_SIZE;
    cuda::barrier<cuda::thread_scope_block>::arrival_token token;

    smem[0][threadIdx.x] = 0.0f;
    smem[1][threadIdx.x] = 0.0f;
    mean[threadIdx.x]    = 0.0f;
    m2[threadIdx.x]      = 0.0f;

    int NUM_LOOPS = stride_b / BLOCK_SIZE;
    float* X_block = X + (stride_b * blockIdx.x);
    float* Y_block = Y + (stride_b * blockIdx.x); 

    if(threadIdx.x == 0){
        init(&bar, 1); 
    }
    __syncthreads();
    if(threadIdx.x == 0){
        cuda::device::memcpy_async_tx(&smem[0][0], X_block, cuda::aligned_size_t<16>(bts), bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, bts);
        bar.wait(cuda::std::move(token));
    }
    __syncthreads();

    
    int NUM = stride_b;
    float count = 0.0f;
    for(int i = 0; i < NUM_LOOPS - 1; i++){
        int next_stage_idx = (i + 1) & 1;
        int stage_idx      = (i & 1);

        if (threadIdx.x == 0) {
            constexpr size_t bytes = sizeof(float) * BLOCK_SIZE;
            cuda::device::memcpy_async_tx(&smem[next_stage_idx][0],
                                          X_block + (i+1) * BLOCK_SIZE,
                                          cuda::aligned_size_t<16>(bytes),
                                          bar);
            token = cuda::device::barrier_arrive_tx(bar, 1, bytes);
            bar.wait(cuda::std::move(token));
        } 
        __syncthreads();
/*
        # else {
        #     token = bar.arrive(1);
        # }

        # bar.wait(cuda::std::move(token));
*/
        float x = smem[stage_idx][threadIdx.x];
        float old_mean = mean[threadIdx.x];
        float delta1   = (x - old_mean);
        count    += 1.0f;
        mean[threadIdx.x] += delta1 / count;
        m2[threadIdx.x]   += delta1 * (x - mean[threadIdx.x]);     
    }
    int stage_idx = (NUM_LOOPS - 1) & 1;
    float x = smem[stage_idx][threadIdx.x];
    float old_mean = mean[threadIdx.x];
    float delta1   = (x - old_mean);
    count    += 1.0f;
    mean[threadIdx.x] += delta1 / count;
    m2[threadIdx.x]   += delta1 * (x - mean[threadIdx.x]);

    //Wellford Combine Reduction 
#pragma unroll
    for(int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2){
        if (threadIdx.x < offset) {
            float delta       = (mean[threadIdx.x + offset] - mean[threadIdx.x]);
            mean[threadIdx.x] = (mean[threadIdx.x] + mean[threadIdx.x + offset]) / 2.0f; 
            m2[threadIdx.x]   += (m2[threadIdx.x + offset] + (delta * delta * count / 2.0f));
        }
        count *= 2.0f;
        __syncthreads(); 
    }

    float var        = m2[0] / (NUM - 1);
    float mean_final = mean[0];
    float std        = rsqrt(var + 1e-5f);


    if(threadIdx.x == 0){
        cuda::device::memcpy_async_tx(&smem[0][0],       X_block, cuda::aligned_size_t<16>(bts), bar);
        cuda::device::memcpy_async_tx(&gamma_smem[0][0], gamma,   cuda::aligned_size_t<16>(bts), bar);
        cuda::device::memcpy_async_tx(&beta_smem[0][0],  beta,    cuda::aligned_size_t<16>(bts), bar);

        token = cuda::device::barrier_arrive_tx(bar, 1, 3 * bts);
    }
    __syncthreads();

    for(int i = 0 ; i < NUM_LOOPS - 1; i++){
        int next_stage_idx = (i + 1) & 1;
        int stage_idx      = (i & 1);
        bar.wait(cuda::std::move(token));  
        if (threadIdx.x == 0) {
            constexpr size_t bytes = sizeof(float) * BLOCK_SIZE;
            cuda::device::memcpy_async_tx(&smem[next_stage_idx][0],       X_block + (i+1) * BLOCK_SIZE, cuda::aligned_size_t<16>(bytes), bar);
            cuda::device::memcpy_async_tx(&gamma_smem[next_stage_idx][0], gamma + (i+1) * BLOCK_SIZE,   cuda::aligned_size_t<16>(bytes), bar);
            cuda::device::memcpy_async_tx(&beta_smem[next_stage_idx][0],  beta + (i+1) * BLOCK_SIZE,    cuda::aligned_size_t<16>(bytes), bar);
            token = cuda::device::barrier_arrive_tx(bar, 1, 3 * bts);
            bar.wait(cuda::std::move(token));
        } 
        __syncthreads();

        float x = smem[stage_idx][threadIdx.x];
        float g = gamma_smem[stage_idx][threadIdx.x];
        float b = beta_smem[stage_idx][threadIdx.x]; 
        float y = ((x - mean_final) * std) * g + b;
        Y_block[threadIdx.x] = y; //GMEM write

        
    }

    x = smem[stage_idx][threadIdx.x];
    float g = gamma_smem[stage_idx][threadIdx.x];
    float b = beta_smem[stage_idx][threadIdx.x]; 
    float y = ((x - mean_final) * std) * g + b;
    Y_block[threadIdx.x] = y; //GMEM write
}

void layer_norm_wellford(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    const size_t STRIDE = F * D1 * D2;
    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t LENGTH = std::max<size_t>(32, next_power_of_2(STRIDE));
    const size_t BLOCK_SIZE = std::min<size_t>(1024, LENGTH);
    dim3 grid(B);
    dim3 block(BLOCK_SIZE);
    switch (BLOCK_SIZE) {
        case 32:
            _layer_norm_wellford<32><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 64:
            _layer_norm_wellford<64><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 128:
            _layer_norm_wellford<128><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 256:
            _layer_norm_wellford<256><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 512:
            _layer_norm_wellford<512><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);
        case 1024:
            _layer_norm_wellford<1024><<<grid, block>>>(Y, X, gamma, beta, B, F, D1, D2, STRIDE);    
    }
    cudaDeviceSynchronize();
}
