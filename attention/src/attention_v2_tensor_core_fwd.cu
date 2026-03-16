#include <cuda_runtime.h>
#include <cuda.h>
#include <utils.h>
#include <map>
#include <stdexcept>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cuda_bf16.h>
#include <math_constants.h>
typedef unsigned int uint;
typedef nv_bfloat16 bf16;

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

template<int BLOCK_K, int DIM, int THREAD_BLOCK_SIZE>
__device__ __forceinline__
void load_gmem_to_smem(const bf16* gmem, const bf16* smem, int tid){
    constexpr int col_bytes = sizeof(bf16) * DIM;
    constexpr int load_size = 16;
    constexpr int num_loads = col_bytes / load_size;
    constexpr int col_stride = load_size / sizeof(bf16);
    constexpr int row_block = THREAD_BLOCK_SIZE / num_loads;
    int tid_x = tid / num_loads;
    int tid_y = tid % num_loads;
    for (int row_block_id = 0; row_block_id < BLOCK_K; row_block_id += row_block){
        int row_offset = tid_x + row_block_id;
        int col_offset = tid_y * col_stride;
        const bf16* smem_ptr = smem + row_offset * DIM + col_offset;
        const bf16* gmem_ptr = gmem + row_offset * DIM + col_offset;
        uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
        asm volatile ("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
    }
}

__device__ __forceinline__
void commit_group(){
    asm volatile ("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__
void wait_group(){
    asm volatile ("cp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__
void wait_all(){
    asm volatile ("cp.async.wait_all;" ::: "memory");
}



template<uint32_t BLOCK_Q, uint32_t BLOCK_KV, uint32_t DIM, uint32_t BLOCK_SIZE>
__global__ void _flash_attention_simple_v2_fwd(
    const bf16* Q, 
    const bf16* K, 
    const bf16* V, 
    bf16* O,
    const uint32_t bs,
    const uint32_t q_len,
    const uint32_t kv_len
)
{
    //Double buffer thus sh*t
    extern __shared__ bf16 smem[];
    bf16 *Q_smem = smem;
    bf16 *K_smem = smem + BLOCK_Q * DIM;
    bf16 *V_smem = K_smem + BLOCK_KV * DIM;
    // __shared__ alignas(16) bf16 Q_smem[BLOCK_Q][DIM];
    // __shared__ alignas(16) bf16 K_smem[BLOCK_KV][DIM];
    // __shared__ alignas(16) bf16 V_smem[BLOCK_KV][DIM];

    constexpr uint32_t MMA_M = 16;
    constexpr uint32_t MMA_N = 8;
    constexpr uint32_t MMA_K = 16;
    constexpr uint32_t NUM_WARPS = (BLOCK_SIZE >> 5);
    constexpr uint32_t WARP_Q = BLOCK_Q / NUM_WARPS;

    const float softmax_scale = rsqrtf((float)DIM);
    const uint32_t batch_idx  = blockIdx.x;
    const uint32_t q_rank     = blockIdx.y;
    const uint32_t warp_id    = (threadIdx.x / 32);
    const uint32_t lane_id    = threadIdx.x % 32;
    
    const bf16* Q_block = Q + batch_idx * (q_len * DIM) + q_rank * BLOCK_Q * DIM;
    bf16* O_block       = O + batch_idx * (q_len * DIM) + q_rank * BLOCK_Q * DIM;
    const bf16* K_base  = K + batch_idx * (kv_len * DIM);
    const bf16* V_base  = V + batch_idx * (kv_len * DIM);
    bf16* Q_warp = Q_smem   + WARP_Q * warp_id * DIM;
    bf16* V_warp = V_smem   + WARP_Q * warp_id * DIM;
    bf16* O_warp = O_block  + WARP_Q * warp_id * DIM;

    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4] = {0};
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2] = {0};
    uint32_t V_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2] = {0};
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][2] = {0};
    
    float O_rmem[WARP_Q / MMA_K][DIM / MMA_N][4] = {0};

    float rowmax_global[WARP_Q / MMA_M][2];
    float logsumexp[WARP_Q / MMA_M][2];
    
    load_gmem_to_smem<BLOCK_Q, DIM, BLOCK_SIZE>(Q_block, Q_smem, threadIdx.x);
    commit_group();
    wait_all();

    // if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
    //     for(int i = 0 ; i < 100; i++){
    //         printf("%.4f ", __bfloat162float(Q_smem[i]));
    //     }
    // }
    // __syncthreads();
    
    for(int row = 0; row < WARP_Q / MMA_M; row += 1){
        for(int col = 0; col < DIM / MMA_K; col += 1){
            int row_offset = (row * MMA_M) + lane_id % 16;
            int col_offset = (col * MMA_K) + (lane_id / 16) * 8 / 2;
            uint32_t q_smem_ptr = __cvta_generic_to_shared(Q_warp + (row_offset * DIM)/2 + col_offset);
            ldmatrix_x4(Q_rmem[row][col], q_smem_ptr);
        }
    }
    
    auto load_k = [&](int idx){
        load_gmem_to_smem<BLOCK_KV, DIM, BLOCK_SIZE>(K_base + idx * BLOCK_KV * DIM, K_smem, threadIdx.x);
    };

    auto load_v = [&](int idx){
        load_gmem_to_smem<BLOCK_KV, DIM, BLOCK_SIZE>(V_base + idx * BLOCK_KV * DIM, V_smem, threadIdx.x);
    };

    for (uint32_t kv_idx = 0; kv_idx < kv_len / BLOCK_KV; kv_idx++){
        float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {0};
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
            printf("starting loop %d\n", kv_idx);
            printf("loop length %d\n", kv_len / BLOCK_KV);
        }
        __syncthreads();
        load_k(kv_idx);
        commit_group();
        wait_group();

        for(int row = 0; row < BLOCK_KV / MMA_N; row++){
            for (int col = 0; col < DIM / MMA_K; col++){
                int row_offset = (row * MMA_N) + lane_id % 8;
                int col_offset = (col * MMA_K) + (lane_id / 8) * 8 / 2;
                uint32_t k_smem_ptr = __cvta_generic_to_shared(K_smem + (row_offset * DIM)/2 + col_offset);
                ldmatrix_x4(K_rmem[row][col], k_smem_ptr);    
            }
        }

    
        
        for(int mma_q = 0; mma_q < WARP_Q / MMA_M; mma_q++){
            for(int mma_k = 0; mma_k < BLOCK_KV / MMA_N; mma_k++){
                for(int mma_dim = 0; mma_dim < DIM / MMA_K; mma_dim++){
                    mma_m16n8k16(
                        Q_rmem[mma_q][mma_dim],
                        K_rmem[mma_k][mma_dim],
                        S_rmem[mma_q][mma_k]
                    );
                }
            }
        }

        for(int mma_q = 0; mma_q < WARP_Q / MMA_M; mma_q++){
            for(int mma_k = 0; mma_k < BLOCK_KV / MMA_N; mma_k++){
                for(int idx = 0; idx < 4; idx++){
                    S_rmem[mma_q][mma_k][idx] *= softmax_scale;
                }
            }
        }

        
        //compute rowmax for each Q
        for(int mma_q = 0; mma_q < WARP_Q / MMA_M; mma_q++){
            float rowmax_tile[2];
            for(int mma_k = 0; mma_k < BLOCK_KV / MMA_N; mma_k++){
                float* reg = S_rmem[mma_q][mma_k];
                rowmax_tile[0] = max(reg[0], reg[1]);
                rowmax_tile[1] = max(reg[2], reg[3]);
                //thread local max
                
                rowmax_tile[0] = max(rowmax_tile[0], __shfl_xor_sync(0xFFFF'FFFF, rowmax_tile[0], 1));
                rowmax_tile[0] = max(rowmax_tile[0], __shfl_xor_sync(0xFFFF'FFFF, rowmax_tile[0], 2));
                rowmax_tile[1] = max(rowmax_tile[1], __shfl_xor_sync(0xFFFF'FFFF, rowmax_tile[1], 1));
                rowmax_tile[1] = max(rowmax_tile[1], __shfl_xor_sync(0xFFFF'FFFF, rowmax_tile[1], 2));
                //current tile rowmax
            }
            rowmax_global[mma_q][0] = max(rowmax_global[mma_q][0], rowmax_tile[0]);
            rowmax_global[mma_q][1] = max(rowmax_global[mma_q][1], rowmax_tile[1]);

            float rescale[2];
            rescale[0] = __expf(rowmax_global[mma_q][0] - rowmax_tile[0]);
            rescale[1] = __expf(rowmax_global[mma_q][1] - rowmax_tile[1]);
            
            for(int mma_d = 0; mma_d < DIM / MMA_N; mma_d++){
                O_rmem[mma_q][mma_d][0] *= rescale[0];
                O_rmem[mma_q][mma_d][1] *= rescale[0];
                O_rmem[mma_q][mma_d][2] *= rescale[1];
                O_rmem[mma_q][mma_d][3] *= rescale[1];
            }
            float rowsum[2] = {};
            for(int mma_k = 0; mma_k < BLOCK_KV / MMA_N; mma_k += 1){
                float* reg = S_rmem[mma_q][mma_k];
                reg[0] = __expf(reg[0] - rowmax_global[mma_q][0]);
                reg[1] = __expf(reg[1] - rowmax_global[mma_q][0]);
                reg[2] = __expf(reg[2] - rowmax_global[mma_q][1]);
                reg[3] = __expf(reg[3] - rowmax_global[mma_q][1]);

                rowsum[0] = reg[0] + reg[1];
                rowsum[1] = reg[2] + reg[3];
                
                nv_bfloat162* P_ptr = reinterpret_cast<nv_bfloat162*>(P_rmem[mma_q][mma_k / 2]);
                P_ptr[(mma_k % 2) * 2]     = __float22bfloat162_rn({reg[0], reg[1]});
                P_ptr[(mma_k % 2) * 2 + 1] = __float22bfloat162_rn({reg[2], reg[3]});
            }


            rowsum[0] = max(rowsum[0], __shfl_xor_sync(0xFFFF'FFFF, rowsum[0], 1));
            rowsum[0] = max(rowsum[0], __shfl_xor_sync(0xFFFF'FFFF, rowsum[0], 2));
            rowsum[1] = max(rowsum[1], __shfl_xor_sync(0xFFFF'FFFF, rowsum[1], 1));
            rowsum[1] = max(rowsum[1], __shfl_xor_sync(0xFFFF'FFFF, rowsum[1], 2));                        

            logsumexp[mma_q][0] = logsumexp[mma_q][0] * rescale[0] + rowsum[0];
            logsumexp[mma_q][1] = logsumexp[mma_q][1] * rescale[1] + rowsum[1];  
        }
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
            printf("loading values idx : %d\n", kv_idx);            
        }        
        load_v(kv_idx);
        commit_group();
        wait_group();
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
            printf("loaded values idx : %d\n", kv_idx); 
        }   
        __syncthreads();
        
        for(int row = 0; row < WARP_Q / MMA_M; row += 1){
            for(int col = 0; col < DIM / MMA_K; col += 1){
                int row_offset = (row * MMA_M) + lane_id % 16;
                int col_offset = ((col * MMA_K) + (lane_id / 16) * 8) / 2;
                uint32_t v_smem_ptr = __cvta_generic_to_shared(V_warp + (row_offset * DIM)/2 + col_offset);
                ldmatrix_x2_trans(V_rmem[row][col], v_smem_ptr);
            }
        }

        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
            printf("done ldmatrixc2 idx : %d\n", kv_idx); 
        }   
        __syncthreads();
        
        for(int mma_q = 0; mma_q < WARP_Q / MMA_M; mma_q++){
            for(int mma_d = 0; mma_d < DIM / MMA_K; mma_d++){
                for(int mma_v = 0; mma_v < BLOCK_KV / MMA_N; mma_v++){
                    mma_m16n8k16(
                        P_rmem[mma_q][mma_v],
                        V_rmem[mma_v][mma_d],
                        O_rmem[mma_q][mma_d]
                    );
                }
            }
        }
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0){
            printf("done mmx idx : %d\n", kv_idx); 
        }   
        __syncthreads();
        
    }
    return ;
    for(int mma_q = 0; mma_q < WARP_Q / MMA_K; mma_q++){
        for(int mma_d = 0; mma_d < DIM / MMA_N; mma_d++){
            float* reg = O_rmem[mma_q][mma_d];
            reg[0] /= logsumexp[mma_q][0];
            reg[1] /= logsumexp[mma_q][0];
            reg[2] /= logsumexp[mma_q][1];
            reg[3] /= logsumexp[mma_q][1];

            int row = (mma_q * MMA_K)/4 + (lane_id / 4);
            int col = (mma_d * MMA_N)/4 + (lane_id % 4);

            reinterpret_cast<nv_bfloat162*>(&O_warp[(row * DIM)])[col] = __float22bfloat162_rn({reg[0], reg[1]});
            reinterpret_cast<nv_bfloat162*>(&O_warp[((row + 8) * DIM)])[col] = __float22bfloat162_rn({reg[2], reg[3]});
        }
    }
    return;
}  

void attention_v2_tc(const bf16* Q, const bf16* K, const bf16* V, bf16* O,
                   const size_t B, const size_t H, const size_t N, const size_t D){

    const uint32_t q_len = N;
    const uint32_t kv_len = N;
    const uint32_t bs = B * H;
    constexpr uint32_t BLOCK_Q = 64;
    constexpr uint32_t BLOCK_KV = 64;
    constexpr uint32_t BLOCKSIZE = 256;
    constexpr uint32_t DIM = 128;
    dim3 grid(bs, N / BLOCK_Q);
    dim3 block(BLOCKSIZE);
    const uint32_t dynamic_bytes = (BLOCK_Q * DIM + 2 * (BLOCK_KV * D)) * sizeof(nv_bfloat16) + 0x1000;
    // std::cout << "Calling tensor core smem : " << dynamic_bytes << "\n";
    switch (D) {
        case 128:
            cudaFuncSetAttribute(_flash_attention_simple_v2_fwd<BLOCK_Q, BLOCK_KV, 128, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
            _flash_attention_simple_v2_fwd<BLOCK_Q, BLOCK_KV, 128, 128><<<grid, block, dynamic_bytes>>>(Q, K, V, O, bs, q_len, kv_len);
            break;
        case 256:
            cudaFuncSetAttribute(_flash_attention_simple_v2_fwd<BLOCK_Q, BLOCK_KV, 256, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_bytes);
            _flash_attention_simple_v2_fwd<BLOCK_Q, BLOCK_KV, 256, 128><<<grid, block, dynamic_bytes>>>(Q, K, V, O, bs, q_len, kv_len);
            break;
    }    
    CUDA_CHECK(cudaDeviceSynchronize());
}
