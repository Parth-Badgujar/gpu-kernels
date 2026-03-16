#include <cuda/ptx>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include "utils.h"

using namespace cooperative_groups;

template<size_t BLOCK_SIZE>
__global__ void _layer_norm_reduce_dsmem_async(float* X, size_t stride_b, float* batch_mean, float* batch_std, size_t B)
{
    __shared__ float mean_smem[BLOCK_SIZE];
    __shared__ float std_smem[BLOCK_SIZE];

    auto cluster = cooperative_groups::this_cluster();

    const int CLUSTER_DIM  = cluster.dim_blocks().x;
    const int BLOCK_EFF    = BLOCK_SIZE * CLUSTER_DIM; 
    int cluster_rank       = cluster.block_rank(); 
    int cluster_idx        = blockIdx.x / CLUSTER_DIM; 
    constexpr int VEC_LOAD = 4;

    if ((size_t)cluster_idx >= B) return;

    float* X_block = X + stride_b * (size_t)cluster_idx + (size_t)(cluster_rank * BLOCK_SIZE * VEC_LOAD);

    const size_t total_vectors = (stride_b + VEC_LOAD - 1) / VEC_LOAD;
    float factor   = 1.0f / (float)stride_b;
    float mean_acc = 0.0f;
    float std_acc  = 0.0f;
    float x[4];

    #pragma unroll
    for(int offset = 0 ; offset < (int)total_vectors; offset += BLOCK_EFF){
        int idx = threadIdx.x + offset;
        if ((size_t)idx >= total_vectors) continue;     
        size_t global_elem = ((size_t)cluster_rank * (size_t)BLOCK_SIZE + (size_t)idx) * (size_t)VEC_LOAD;
        if (global_elem + (VEC_LOAD - 1) < stride_b) {
            reinterpret_cast<float4*>(x)[0] = reinterpret_cast<float4*>(X_block)[idx];
            #pragma unroll
            for(int j = 0; j < VEC_LOAD; j++) mean_acc += x[j];
            #pragma unroll
            for(int j = 0; j < VEC_LOAD; j++) std_acc  += (x[j] * x[j]);
        } else {
            int base_in_block = idx * VEC_LOAD;
            #pragma unroll
            for(int j = 0; j < VEC_LOAD; j++){
                size_t abs_idx = global_elem + (size_t)j;
                if (abs_idx < stride_b) {
                    float val = X_block[base_in_block + j];
                    mean_acc += val;
                    std_acc  += val * val;
                }
            }
        }
    }

    mean_smem[threadIdx.x] = mean_acc * factor;
    std_smem[threadIdx.x]  = std_acc * factor;

    #pragma unroll              
    for(int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
        __syncthreads(); 
        float sval = std_smem[threadIdx.x ^ stride]; 
        float mval = mean_smem[threadIdx.x ^ stride];
        __syncthreads();
        std_smem[threadIdx.x]  += sval;
        mean_smem[threadIdx.x] += mval;
    }

    cluster.sync();    
    cluster_all_reduce<float>(&mean_smem[threadIdx.x], cluster_rank, cluster);
    cluster_all_reduce<float>(&std_smem[threadIdx.x],  cluster_rank, cluster);
    cluster.sync();

    float mean = mean_smem[0];
    float std  = rsqrtf((std_smem[0] - (mean * mean)) + 1e-5f);

    if(threadIdx.x == 0 && cluster_rank == 0){
        batch_mean[cluster_idx] = mean;
    }
    if(threadIdx.x == 1 && cluster_rank == 0){
        batch_std[cluster_idx]  = std;
    }
}

template<size_t BLOCK_SIZE>
__global__ void _layer_norm_pointwise_dsmem_async(float* Y, float* X, float* gamma, float* beta, size_t stride_b, float* batch_mean, float* batch_std, size_t B)
{
    __shared__ float mean_smem[BLOCK_SIZE];
    __shared__ float std_smem[BLOCK_SIZE];

    __shared__ cuda::barrier<cuda::thread_scope_block> barrier[2][2]; 
    cuda::barrier<cuda::thread_scope_block>::arrival_token token[2][2];
    if(threadIdx.x < 2){
        new (&barrier[threadIdx.x][0]) cuda::barrier<cuda::thread_scope_block>(blockDim.x);
        new (&barrier[threadIdx.x][1]) cuda::barrier<cuda::thread_scope_block>(blockDim.x);
    }
    __syncthreads();

    constexpr int VEC_LOAD = 4;

    int threads_per_batch = (BLOCK_SIZE + (int)B - 1) / (int)B; 
    int batch_idx         = threadIdx.x / threads_per_batch; 
    int thread_idx        = threadIdx.x % threads_per_batch; 
    int block_offset      = (threads_per_batch * blockIdx.x);
    int increment         = (threads_per_batch * gridDim.x);

    if (batch_idx >= (int)B) return;

    float* X_block = X + (size_t)batch_idx * (size_t)stride_b;
    float* Y_block = Y + (size_t)batch_idx * (size_t)stride_b;
    X_block += block_offset;
    Y_block += block_offset;
    float* gamma_base = gamma + block_offset;
    float* beta_base  = beta  + block_offset;

    float* gamma_smem = mean_smem;
    float* beta_smem  = std_smem;

    auto size_aligned = cuda::aligned_size_t<16>(threads_per_batch * sizeof(float));
    float mean = batch_mean[batch_idx];
    float std  = batch_std[batch_idx];

    if(threadIdx.x == 0){
        cuda::device::memcpy_async_tx(gamma_smem, gamma_base, size_aligned, barrier[0][0]);
        cuda::device::memcpy_async_tx(beta_smem,  beta_base,  size_aligned, barrier[0][1]);
        token[0][0] = cuda::device::barrier_arrive_tx(barrier[0][0], 1, threads_per_batch * sizeof(float));
        token[0][1] = cuda::device::barrier_arrive_tx(barrier[0][1], 1, threads_per_batch * sizeof(float));
    }
    else {
        token[0][0] = barrier[0][0].arrive();
        token[0][1] = barrier[0][1].arrive();
    }

    int num_iters = (stride_b + increment - 1) / increment; 
    for(int offset = 0; offset < num_iters - 1; offset += 1){
        int stage_idx      = (offset & 1);
        int next_stage_idx = (offset + 1) & 1;
        int curr_block_idx = offset * increment;

        if(threadIdx.x == 0){
            cuda::device::memcpy_async_tx(gamma_smem + next_stage_idx * threads_per_batch, &gamma_base[curr_block_idx], size_aligned, barrier[next_stage_idx][0]);    
            cuda::device::memcpy_async_tx(beta_smem  + next_stage_idx * threads_per_batch, &beta_base[curr_block_idx],  size_aligned, barrier[next_stage_idx][1]);
            token[next_stage_idx][0] = cuda::device::barrier_arrive_tx(barrier[next_stage_idx][0], 1, threads_per_batch * sizeof(float));
            token[next_stage_idx][1] = cuda::device::barrier_arrive_tx(barrier[next_stage_idx][1], 1, threads_per_batch * sizeof(float));
        }
        else {
            token[next_stage_idx][0] = barrier[next_stage_idx][0].arrive();
            token[next_stage_idx][1] = barrier[next_stage_idx][1].arrive();
        }

        barrier[stage_idx][0].wait(cuda::std::move(token[stage_idx][0]));
        barrier[stage_idx][1].wait(cuda::std::move(token[stage_idx][1]));
        
        float x_loc = X_block[thread_idx + curr_block_idx];
        float g = gamma_smem[thread_idx + stage_idx * threads_per_batch];
        float b = beta_smem[thread_idx + stage_idx * threads_per_batch];
        x_loc = ((x_loc - mean) * std) * g + b; 
        Y_block[thread_idx + curr_block_idx] = x_loc;
    }

    int offset         = num_iters - 1;
    int stage_idx      = (offset & 1);
    int curr_block_idx = offset * increment;

    int global_idx = offset * increment + thread_idx; 

    barrier[stage_idx][0].wait(cuda::std::move(token[stage_idx][0]));
    barrier[stage_idx][1].wait(cuda::std::move(token[stage_idx][1]));
    
    if ((size_t)global_idx < stride_b){
        float x_loc = X_block[thread_idx + offset * increment];
        float g = gamma_smem[thread_idx + stage_idx * threads_per_batch];
        float b = beta_smem[thread_idx + stage_idx * threads_per_batch];
        x_loc = ((x_loc - mean) * std) * g + b; 
        Y_block[thread_idx + offset * increment] = x_loc;
    }
}

void layer_norm_simple_dsmem_async_v1(float *X, float *Y, float *gamma, float *beta, size_t B, size_t F, size_t D1, size_t D2){
    const size_t MAX_CLUSTERS   = 8; //SM_120 :(
    const size_t MAX_BLOCK_SIZE = 1024;
    const size_t STRIDE         = F * D1 * D2;
    const size_t VECTOR_SIZE    = (STRIDE + 3) / 4; //Vectorized loads
    const size_t LENGTH         = next_power_of_2(VECTOR_SIZE);
    const size_t BLOCK_SIZE     = std::min<size_t>(MAX_BLOCK_SIZE, LENGTH);
    const size_t NUM_CLUSTERS   = std::min<size_t>(MAX_CLUSTERS, (LENGTH + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    const size_t GRID_SIZE_R    = B * NUM_CLUSTERS;

    dim3 blockDim(BLOCK_SIZE);

    float *batch_mean, *batch_std;
    cudaMalloc(&batch_mean, sizeof(float) * B * 2);
    batch_std = &batch_mean[B];

    // ---------------------------
    // reduction launch (clustered)
    // ---------------------------
    {
        dim3 gridDim(GRID_SIZE_R);

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
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<32>, X, STRIDE, batch_mean, batch_std, B);
                break;
            case 64:
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<64>, X, STRIDE, batch_mean, batch_std, B);
                break;
            case 128:
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<128>, X, STRIDE, batch_mean, batch_std, B);
                break;
            case 256:
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<256>, X, STRIDE, batch_mean, batch_std, B);
                break;
            case 512:
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<512>, X, STRIDE, batch_mean, batch_std, B);
                break;
            case 1024:
                cudaLaunchKernelEx(&config, _layer_norm_reduce_dsmem_async<1024>, X, STRIDE, batch_mean, batch_std, B);
                break;
        }
    }

    // ---------------------------
    // pointwise/apply launch (non-cluster)
    // ---------------------------
    {
        // match kernel’s internal partitioning:
        cudaDeviceSynchronize();
        int threads_per_batch = (int)((BLOCK_SIZE + B - 1) / B);
        int grid_x            = (int)((STRIDE + threads_per_batch - 1) / threads_per_batch);
        dim3 gridDim(grid_x);

        cudaLaunchConfig_t config = {0};
        config.gridDim  = gridDim;
        config.blockDim = blockDim;
        config.dynamicSmemBytes = 0; 
        config.stream = 0;
        config.numAttrs = 0;
        config.attrs = nullptr;

        switch(BLOCK_SIZE){
            case 32:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<32>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
            case 64:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<64>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
            case 128:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<128>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
            case 256:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<256>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
            case 512:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<512>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
            case 1024:
                cudaLaunchKernelEx(&config, _layer_norm_pointwise_dsmem_async<1024>, Y, X, gamma, beta, STRIDE, batch_mean, batch_std, B);
                break;
        }
    }

    cudaDeviceSynchronize();
    cudaFree(batch_mean);
}
