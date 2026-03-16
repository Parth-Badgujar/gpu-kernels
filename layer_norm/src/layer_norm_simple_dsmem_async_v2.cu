#include <cooperative_groups.h>
#include <cuda/barrier>
#include "utils.h"

template<size_t BLOCK_SIZE>
__global__ void _layer_norm_simple_dsmem_async_v2(float* Y, float* X, float* gamma, float* beta, size_t stride_b)
{
    __shared__ float4 smem[2][BLOCK_SIZE];
    float* mean_smem = reinterpret_cast<float*>(smem[0]);
    float* std_smem  = reinterpret_cast<float*>(smem[1]);
    constexpr int VECTOR_LOAD = 4;
    auto cluster = cooperative_groups::this_cluster();
    const int CLUSTER_DIM  = cluster.dim_blocks().x;
    const int BLOCK_EFF    = BLOCK_SIZE * CLUSTER_DIM;
    const int cluster_rank       = cluster.block_rank();
    const int cluster_idx        = blockIdx.x / CLUSTER_DIM;
    const int batch_offset    = stride_b * cluster_idx;
    const int cluster_offset  = cluster_rank * BLOCK_SIZE * VECTOR_LOAD;
    float4* X_block     = reinterpret_cast<float4*>(X + batch_offset + cluster_offset);
    float4* Y_block           = reinterpret_cast<float4*>(Y + batch_offset + cluster_offset);
    const float4* gamma_block = reinterpret_cast<float4*>(gamma + cluster_offset);
    const float4* beta_block  = reinterpret_cast<float4*>(beta + cluster_offset);
	const int total_vectors = (stride_b + VECTOR_LOAD - 1) / VECTOR_LOAD; 
    const float factor   = 1.0f / (float)stride_b;
    float mean_acc = 0.0f;
    float std_acc  = 0.0f;
    int parity[2] = {1, 1};

    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];
    cuda::barrier<cuda::thread_scope_block>::arrival_token token[2];
    auto mem = cuda::aligned_size_t<16>(BLOCK_SIZE * 4);
    if (threadIdx.x == 0){
        init(&bar[0], 1);
        init(&bar[1], 1);
        token[0] = cuda::device::barrier_arrive_tx(bar[0], 1, mem);
        cuda::device::memcpy_async_tx(
            reinterpret_cast<float*>(smem[0]), 
            reinterpret_cast<float*>(X_block), 
            mem, 
            bar[0]
        );
    }
    float4 x;
    int counter = 0;
    int num_iters = (total_vectors + BLOCK_EFF - 1) / BLOCK_EFF;
    int curr_idx, stage, next_stage;
    for(curr_idx = 0; curr_idx < num_iters - 1; curr_idx += 1){
        stage = curr_idx & 1;
        next_stage = (curr_idx + 1) & 1;
        if(threadIdx.x == 0){
            token[next_stage] = cuda::device::barrier_arrive_tx(bar[next_stage], 1, mem);
            cuda::device::memcpy_async_tx(
                reinterpret_cast<float*>(smem[next_stage]), 
                reinterpret_cast<float*>(X_block + (curr_idx + 1) * BLOCK_EFF), 
                mem, 
                bar[next_stage]
            );
            // bar[stage].wait(cuda::std::move(token[stage]));
        }
        parity[stage] ^= 1;
        __syncthreads();
        cuda::ptx::mbarrier_try_wait_parity(
           cuda::ptx::sem_relaxed,                    // or sem_relaxed if you know what you're doing
           cuda::ptx::scope_cta,
           cuda::device::barrier_native_handle(bar[stage]),
           parity[stage]
        );
        // cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar[stage]), parity[stage]);
        x = smem[stage][threadIdx.x];
        mean_acc += x.x;
        mean_acc += x.y;
        mean_acc += x.z;
        mean_acc += x.w;
        std_acc  += (x.x * x.x);
        std_acc  += (x.y * x.y);
        std_acc  += (x.z * x.z);
        std_acc  += (x.w * x.w);        
    }
    stage = (num_iters - 1) & 1;
    // if (threadIdx.x == 0){
    //     bar[stage].wait(cuda::std::move(token[stage]));    
    // }
    // __syncthreads();
    parity[stage] ^= 1;
    __syncthreads();
    cuda::ptx::mbarrier_try_wait_parity(
           cuda::ptx::sem_acquire,                    // or sem_relaxed if you know what you're doing
           cuda::ptx::scope_cta,
           cuda::device::barrier_native_handle(bar[stage]),
           parity[stage]
        );
    x = smem[stage][threadIdx.x];
    mean_acc += x.x;
    mean_acc += x.y;
    mean_acc += x.z;
    mean_acc += x.w;
    std_acc  += (x.x * x.x);
    std_acc  += (x.y * x.y);
    std_acc  += (x.z * x.z);
    std_acc  += (x.w * x.w);        
    __syncthreads();
    
    mean_smem[threadIdx.x] = mean_acc * factor;
    std_smem[threadIdx.x]  = std_acc * factor;

    
    block_reduce<float, BLOCK_SIZE>(mean_smem);
    block_reduce<float, BLOCK_SIZE>(std_smem);
    
    std_smem[threadIdx.x]  = std_smem[0];
    mean_smem[threadIdx.x] = mean_smem[0];

    if (CLUSTER_DIM > 1){
        cluster.sync();
        cluster_all_reduce<float>(&mean_smem[threadIdx.x], cluster_rank, cluster);
        cluster_all_reduce<float>(&std_smem[threadIdx.x],  cluster_rank, cluster);
    }
    
    float mean = mean_smem[0];
    float std  = rsqrtf((std_smem[0] - (mean * mean)) + 1e-5f);
    
    float4 g;
    float4 b;
    float4 y;
    
    #pragma unroll
    for(int offset = 0; offset < (int)total_vectors; offset += BLOCK_EFF){
        int idx = threadIdx.x + offset;
        if ((size_t)idx >= total_vectors) continue; 
        size_t global_elem = ((size_t)cluster_rank * (size_t)BLOCK_SIZE + (size_t)idx) * (size_t)VECTOR_LOAD;
        x = X_block[idx];
        g = gamma_block[idx];
        b = beta_block[idx];
        y.x = ((x.x - mean) * std) * g.x + b.x;
        y.y = ((x.y - mean) * std) * g.y + b.y;
        y.z = ((x.z - mean) * std) * g.z + b.z;
        y.w = ((x.w - mean) * std) * g.w + b.w;
        Y_block[idx] = y;
    }
}

void layer_norm_simple_dsmem_async_v2(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    auto max = std::max<size_t>; 
    auto min = std::min<size_t>; 
    const size_t MAX_CLUSTERS   = 8; //SM_120 :(
    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t STRIDE         = F * D1 * D2;
    const size_t VECTOR_SIZE    = (STRIDE + 3) / 4; //Vectorized loads
    // const size_t VECTOR_SIZE    = (STRIDE + 3) / 2; //Vectorized loads
    // const size_t VECTOR_SIZE    = (STRIDE + 3); //Vectorized loads
    const size_t LENGTH         = next_power_of_2(VECTOR_SIZE);
    const size_t BLOCK_SIZE     = max(32, min(MAX_BLOCK_SIZE, LENGTH));
    const size_t NUM_CLUSTERS   = min(MAX_CLUSTERS, (LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const size_t GRID_SIZE      = B * NUM_CLUSTERS;

    dim3 gridDim(GRID_SIZE);
    dim3 blockDim(BLOCK_SIZE);

    cudaLaunchConfig_t config = {0};
    config.gridDim  = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0; 
    config.stream = 0;
    cudaLaunchAttribute attribute[1]; 
    attribute[0].id = cudaLaunchAttributeClusterDimension; 
    attribute[0].val.clusterDim.x = NUM_CLUSTERS;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.numAttrs = 1;
    config.attrs = attribute;

    switch(BLOCK_SIZE){
        case 32:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<32>, Y, X, gamma, beta, STRIDE);
            break;
        case 64:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<64>, Y, X, gamma, beta, STRIDE);
            break;
        case 128:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<128>, Y, X, gamma, beta, STRIDE);
            break;
        case 256:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<256>, Y, X, gamma, beta, STRIDE);
            break;
        case 512:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<512>, Y, X, gamma, beta, STRIDE);
            break;
        case 1024:
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem_async_v2<1024>, Y, X, gamma, beta, STRIDE);
            break;
        
    }
    cudaDeviceSynchronize();
}
