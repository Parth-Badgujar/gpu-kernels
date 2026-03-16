#include <cooperative_groups.h>


static size_t next_power_of_2(size_t num){
    int cntr = 0;
    int one_bit = 0;
    while(num > 0){
        one_bit += (num & 1);
        num >>= 1;
        cntr += 1;
    }
    if (one_bit > 1){
        return (1 << cntr);
    }
    else {
        return (1 << (cntr+1));
    }
}


template <class T>
__device__ __forceinline__
void cluster_all_reduce(T* data, int cluster_rank, cooperative_groups::cluster_group& cluster){
    int cluster_dim = cluster.dim_blocks().x;
    #pragma unroll
    for(int offset = 1; offset < cluster_dim; offset *= 2){
        T* dsmem_data = cluster.map_shared_rank(data, cluster_rank ^ offset); //Butterfly ALL-REDUCE
        cluster.sync();
        float val = dsmem_data[0];
        cluster.sync();
        data[0] += val;
        
    }
}

template <class T, size_t BLOCK_SIZE>
__device__ __forceinline__
void block_reduce(T* smem1){
    #pragma unroll              
    for(int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1){
        if (threadIdx.x < stride){
            T sval = smem1[threadIdx.x + stride]; 
            smem1[threadIdx.x]  += sval;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32){
        auto FULL_MASK = __activemask();
        T val1 = smem1[threadIdx.x];
        val1 += __shfl_down_sync(FULL_MASK, val1, 16);
        val1 += __shfl_down_sync(FULL_MASK, val1, 8);
        val1 += __shfl_down_sync(FULL_MASK, val1, 4);
        val1 += __shfl_down_sync(FULL_MASK, val1, 2);
        val1 += __shfl_down_sync(FULL_MASK, val1, 1);
        if(threadIdx.x == 0){
            smem1[0] = val1;
        }    
    }
    __syncthreads();
}


template <class T, size_t BLOCK_SIZE>
__device__ __forceinline__
void double_block_reduce(T* smem1, T* smem2){
    static_assert(BLOCK_SIZE >= 64);
    #pragma unroll              
    for(int stride = BLOCK_SIZE / 2; stride > 16; stride >>= 1){
        if (threadIdx.x < stride){
            T sval1 = smem1[threadIdx.x + stride]; 
            smem1[threadIdx.x]  += sval1;
        }
        if (threadIdx.x >= (BLOCK_SIZE - stride)){
            T sval2 = smem2[threadIdx.x - stride];
            smem2[threadIdx.x] += sval2;
        }
        __syncthreads();
    }
    if (threadIdx.x < 32){
        auto FULL_MASK = __activemask();
        T val1 = smem1[threadIdx.x];
        val1 += __shfl_down_sync(FULL_MASK, val1, 16);
        val1 += __shfl_down_sync(FULL_MASK, val1, 8);
        val1 += __shfl_down_sync(FULL_MASK, val1, 4);
        val1 += __shfl_down_sync(FULL_MASK, val1, 2);
        val1 += __shfl_down_sync(FULL_MASK, val1, 1);
        if((threadIdx.x & 31) == 0){
            smem1[0] = val1;
        }    
    }
    if (threadIdx.x >= (BLOCK_SIZE - 32)){
        auto FULL_MASK = __activemask();
        T val2 = smem2[threadIdx.x];
        val2 += __shfl_down_sync(FULL_MASK, val2, 16);
        val2 += __shfl_down_sync(FULL_MASK, val2, 8);
        val2 += __shfl_down_sync(FULL_MASK, val2, 4);
        val2 += __shfl_down_sync(FULL_MASK, val2, 2);
        val2 += __shfl_down_sync(FULL_MASK, val2, 1);
        if((threadIdx.x & 31) == 0){
            smem2[0] = val2;
        }
    }
    __syncthreads();
}

template<size_t BLOCK_SIZE>
__global__ void _layer_norm_simple_dsmem(float* __restrict__ Y, float* __restrict__ X, float* __restrict__ gamma, float* __restrict__ beta, size_t stride_b)
{
    // extern __shared__ float ext_smem[];
    // float* mean_smem = ext_smem;
    // float* std_smem = ext_smem;
    __shared__ float mean_smem[BLOCK_SIZE];
    __shared__ float std_smem[BLOCK_SIZE];
    // float* std_smem = mean_smem;
    constexpr int VECTOR_LOAD = 4;
    constexpr int STAGE_STRIDE = BLOCK_SIZE;
    auto cluster = cooperative_groups::this_cluster();
    const int CLUSTER_DIM     = cluster.dim_blocks().x;
    const int BLOCK_EFF       = BLOCK_SIZE * CLUSTER_DIM;
    const int cluster_rank    = cluster.block_rank();
    const int cluster_idx     = blockIdx.x / CLUSTER_DIM;
    const int batch_offset    = stride_b * cluster_idx;
    const int cluster_offset  = cluster_rank * BLOCK_SIZE * VECTOR_LOAD;
    const float4* X_block     = reinterpret_cast<float4*>(X + batch_offset + cluster_offset);
    float4* Y_block           = reinterpret_cast<float4*>(Y + batch_offset + cluster_offset);
    const float4* gamma_block = reinterpret_cast<float4*>(gamma + cluster_offset);
    const float4* beta_block  = reinterpret_cast<float4*>(beta + cluster_offset);
	const int total_vectors = (stride_b + VECTOR_LOAD - 1) >> 2; 
    const float factor   = 1.0f / (float)stride_b;
    
    float mean = 0.0f;
    float std  = 0.0f;
    float4 x;
    #pragma unroll
    for(int offset = 0 ; offset < total_vectors; offset += BLOCK_EFF){
        int idx = threadIdx.x + offset;
        int global_idx = (cluster_rank * BLOCK_SIZE + idx);
        if (global_idx >= total_vectors) continue;
        x = X_block[idx];
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
    std_smem[threadIdx.x]  = std;
    __syncthreads();
    if constexpr (BLOCK_SIZE == 32){
        block_reduce<float, BLOCK_SIZE>(mean_smem);
        block_reduce<float, BLOCK_SIZE>(std_smem);    
    }
    else{
        double_block_reduce<float, BLOCK_SIZE>(mean_smem, std_smem);    
    }
    std_smem[threadIdx.x]  = std_smem[0];
    mean_smem[threadIdx.x] = mean_smem[0];

    cluster_all_reduce<float>(&mean_smem[threadIdx.x], cluster_rank, cluster);
    cluster_all_reduce<float>(&std_smem[threadIdx.x],  cluster_rank, cluster);
    // cluster.sync();
    
    // mean_smem[threadIdx.x] = mean * factor;
    // block_reduce<float, BLOCK_SIZE>(mean_smem);
    // mean_smem[threadIdx.x] = mean_smem[0];
    // cluster.sync();    
    // cluster_all_reduce<float>(&mean_smem[threadIdx.x], cluster_rank, cluster);
    // cluster.sync();
    // std_smem[threadIdx.x] = std * factor;
    // block_reduce<float, BLOCK_SIZE>(std_smem);
    // std_smem[threadIdx.x] = std_smem[0];
    // cluster.sync();    
    // cluster_all_reduce<float>(&std_smem[threadIdx.x], cluster_rank, cluster);
    // cluster.sync();
    mean = mean_smem[0] * factor;
    std  = std_smem[0] * factor;
    
    std  = rsqrtf((std - (mean * mean)) + 1e-5f);
    
    float4 g;
    float4 b;
    float4 y;
    int stage_counter = 0;
    int stage, next_stage;
    for(int offset = 0; offset < (int)total_vectors; offset += BLOCK_EFF){
        int idx = threadIdx.x + offset;
        int global_idx = (cluster_rank * BLOCK_SIZE + idx);
        if (global_idx >= total_vectors) continue;
        x = X_block[idx];
        g = gamma_block[idx];
        b = beta_block[idx];
        y.x = ((x.x - mean) * std) * g.x + b.x;
        y.y = ((x.y - mean) * std) * g.y + b.y;
        y.z = ((x.z - mean) * std) * g.z + b.z;
        y.w = ((x.w - mean) * std) * g.w + b.w;
        // stage = (stage_counter % 2);
        // stage_counter++;
        // asm volatile("cp.async.bulk.wait_group 2;");
        // smem[stage * BLOCK_SIZE + threadIdx.x] = y;
        // __syncwarp();
        
        Y_block[idx] = y;
        // Y_block[idx] = smem[stage * BLOCK_SIZE + threadIdx.x];
        // asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        // if(lane_id == 0){
        //     async_store_tma(reinterpret_cast<float*>(&Y_block[idx]), reinterpret_cast<float*>(&smem[1 * BLOCK_SIZE + 32 * warp_id]), 4 * 32 * sizeof(float));    
        // }
        
        // asm volatile("cp.async.bulk.commit_group;");    
        // __syncthreads();
        
    }
    // asm volatile("cp.async.bulk.wait_group 0;");
}


void layer_norm_simple_dsmem(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    auto max = std::max<size_t>; 
    auto min = std::min<size_t>; 
    const size_t MAX_CLUSTERS   = 8; //SM_120 :(
    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t STRIDE         = F * D1 * D2;
    const size_t VECTOR_SIZE    = (STRIDE + 3) / 4; //Vectorized loads
    const size_t LENGTH         = next_power_of_2(VECTOR_SIZE);
    const size_t BLOCK_SIZE     = max(32, min(MAX_BLOCK_SIZE, LENGTH));
    const size_t NUM_CLUSTERS   = min(MAX_CLUSTERS, (LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const size_t GRID_SIZE      = B * NUM_CLUSTERS;
    // const size_t DSMEM_BYTES    = BLOCK_SIZE * 4 + 4 * 4 * BLOCK_SIZE * 3;
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
            // // cudaFuncSetAttribute(_layer_norm_simple_dsmem<32>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<32>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<32>, Y, X, gamma, beta, STRIDE);
            break;
        case 64:
            // cudaFuncSetAttribute(_layer_norm_simple_dsmem<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<64>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<64>, Y, X, gamma, beta, STRIDE);
            break;
        case 128:
            // cudaFuncSetAttribute(_layer_norm_simple_dsmem<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<128>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<128>, Y, X, gamma, beta, STRIDE);
            break;
        case 256:
            // cudaFuncSetAttribute(_layer_norm_simple_dsmem<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<256>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<256>, Y, X, gamma, beta, STRIDE);
            break;
        case 512:
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<512>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            // cudaFuncSetAttribute(_layer_norm_simple_dsmem<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<512>, Y, X, gamma, beta, STRIDE);
            break;
        case 1024:
            // cudaFuncSetAttribute(_layer_norm_simple_dsmem<1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM_BYTES);
            cudaFuncSetAttribute(_layer_norm_simple_dsmem<1024>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
            cudaLaunchKernelEx(&config, _layer_norm_simple_dsmem<1024>, Y, X, gamma, beta, STRIDE);
            break;        
    }
    cudaDeviceSynchronize();
}
