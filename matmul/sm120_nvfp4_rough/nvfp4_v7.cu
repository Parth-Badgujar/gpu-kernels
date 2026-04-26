#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <torch/torch.h>
#include "../../include/mma_sync.cuh"
#include "../../include/tma.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/fence.cuh"
#include "../../include/others.cuh"
#include "../../include/ldmatrix.cuh"

#define MMA_K 64
#define MMA_M 16
#define MMA_N 8

__device__ __forceinline__ int2 get_grid_coords(int bid, int M_BLOCKS, int N_BLOCKS) {
    constexpr int TILE_SIZE = 4; 
    int tiles_per_row = N_BLOCKS / TILE_SIZE;
    int super_tile_id = bid / (TILE_SIZE * TILE_SIZE);
    int local_id = bid % (TILE_SIZE * TILE_SIZE);
    int super_tile_row = super_tile_id / tiles_per_row;
    int super_tile_col = super_tile_id % tiles_per_row;
    int local_row = local_id / TILE_SIZE;
    int local_col = local_id % TILE_SIZE;
    int m_block = super_tile_row * TILE_SIZE + local_row;
    int n_block = super_tile_col * TILE_SIZE + local_col;
    if (m_block >= M_BLOCKS || n_block >= N_BLOCKS) {
        m_block = bid % M_BLOCKS;
        n_block = bid / M_BLOCKS;
    }
    return make_int2(m_block, n_block);
}

template<int BLOCK_M = 128, int BLOCK_N = 128, int BLOCK_K = 64, int NUM_STAGES = 1>
__global__ void _nvfp4_gemm_v7(
    const uint8_t* __restrict__ gSFA,
    const uint8_t* __restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    float* __restrict__ gC,
    int M, int N, int K
){
    constexpr int sA_mem   = (BLOCK_M * BLOCK_K / 2);
    constexpr int sB_mem   = (BLOCK_N * BLOCK_K / 2);
    constexpr int sSFA_mem = (BLOCK_M * BLOCK_K / 16);
    constexpr int sSFB_mem = (BLOCK_N * BLOCK_K / 16);
    constexpr int STAGE_SIZE = sA_mem + sB_mem + sSFA_mem + sSFB_mem;
    constexpr int WARP_M = BLOCK_M / 4;
    constexpr int WARP_N = BLOCK_N / 2;
    __shared__ alignas(1024) uint8_t  smem[STAGE_SIZE * NUM_STAGES];
    __shared__ alignas(128) uint64_t load_mbar_ptr[NUM_STAGES];
    __shared__ alignas(128) uint64_t compute_mbar_ptr[NUM_STAGES];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int quad_id = lane_id / 4;
    // int load_phase[NUM_STAGES] = {0};
    // int compute_phase[NUM_STAGES] = {0};
    // 1 register can hold up to 32 stages. 0 local memory.
    uint32_t load_phases = 0;    
    uint32_t compute_phases = 0;
    int lane_id_in_quad = lane_id % 4;
    int wid_x = warp_id / 2;
    int wid_y = warp_id % 2;

    uint32_t smem_base = __cvta_generic_to_shared(smem);
    uint32_t compute_mbar = __cvta_generic_to_shared(compute_mbar_ptr);
    uint32_t load_mbar = __cvta_generic_to_shared(load_mbar_ptr);
    if (warp_id == 0 && elect_sync()) {
        for(int i = 0; i < NUM_STAGES; i++){
            mbarrier_init(compute_mbar + i * 8, 32 * 8);
            mbarrier_init(load_mbar + i * 8, 1);
        }
        fence_mbarrier_init();
        for(int i = 0; i < NUM_STAGES; i++)
            mbarrier_arrive_count(compute_mbar + i * 8, 32 * 8);
    }
    __syncthreads();

    uint32_t sA   = smem_base;
    uint32_t sB   = sA   + sA_mem;
    uint32_t sSFA = sB   + sB_mem;
    uint32_t sSFB = sSFA + sSFA_mem;
    float zero[4] = {0.0f};

    float    rC[WARP_M / MMA_M][WARP_N / MMA_N][4];
    uint32_t rA[2][WARP_M / MMA_M][4];
    uint32_t rB[2][WARP_N / MMA_N][2];
    uint32_t rSFA[2][(WARP_M / MMA_M) / 2];
    uint32_t rSFB[2][(WARP_N / MMA_N) / 4];

    auto load_smem = [&](int iter_k, int stage_id, int off_n, int off_m){
        tma_load_2d(sA + stage_id * STAGE_SIZE,   &tmap_A, load_mbar + stage_id * 8, iter_k * BLOCK_K, off_m);
        tma_load_2d(sB + stage_id * STAGE_SIZE,   &tmap_B, load_mbar + stage_id * 8, iter_k * BLOCK_K, off_n);
        tma_load_flat(sSFA + stage_id * STAGE_SIZE, gSFA + off_m * (K / 16) + iter_k * (BLOCK_M * BLOCK_K / 16), sSFA_mem, load_mbar + stage_id * 8, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        tma_load_flat(sSFB + stage_id * STAGE_SIZE, gSFB + off_n * (K / 16) + iter_k * (BLOCK_N * BLOCK_K / 16), sSFB_mem, load_mbar + stage_id * 8, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };

    auto load_regs = [&](int mma_k, int stage_id, int reg_phase){
        uint32_t* sfa_ptr = (uint32_t*)__cvta_shared_to_generic(sSFA + stage_id * STAGE_SIZE);
        uint32_t* sfb_ptr = (uint32_t*)__cvta_shared_to_generic(sSFB + stage_id * STAGE_SIZE);
        uint32_t sA_curr = sA + stage_id * STAGE_SIZE;
        uint32_t sB_curr = sB + stage_id * STAGE_SIZE;
        int row_sf = (quad_id + lane_id_in_quad * 8);
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m += 2){
            int col = wid_x * 2 + (mma_m / 2);
            rSFA[reg_phase][mma_m / 2] = sfa_ptr[(mma_k << 7) + col + (row_sf << 2)];
        }
        #pragma unroll
        for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n += 4){
            int col = wid_y * 2 + (mma_n / 4);
            rSFB[reg_phase][mma_n / 4] = sfb_ptr[(mma_k << 7) + col + (row_sf << 2)];
        }

        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            int col = mma_k * (MMA_K / 2) + (lane_id / 16) * 16;
            int row = (mma_m * MMA_M + wid_x * WARP_M + (lane_id % 16));
            ldmatrix_m8n8_x4_b16(rA[reg_phase][mma_m], swizzle<3, 4, 3>(sA_curr + row * (BLOCK_K / 2) + col));
        }
        #pragma unroll
        for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
            int col = mma_k * (MMA_K / 2) + (lane_id / 8) * 16;
            int row = (mma_n * MMA_N + wid_y * WARP_N + (lane_id % 8));
            ldmatrix_m8n8_x2_b16(rB[reg_phase][mma_n], swizzle<3, 4, 3>(sB_curr + row * (BLOCK_K / 2) + col));
        }
    };

    auto compute = [&](int reg_phase, bool acc){
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
                mma_m16n8k64_mxf4nvf4_4x_ue4m3(
                    rC[mma_m][mma_n],
                    rA[reg_phase][mma_m],
                    rB[reg_phase][mma_n],
                    acc ? rC[mma_m][mma_n] : zero,
                    rSFA[reg_phase][mma_m / 2],
                    rSFB[reg_phase][mma_n / 4],
                    0,
                    mma_m % 2,
                    0,
                    mma_n % 4
                );
            }
        }
    };

    auto store = [&](float* blockC){
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            int current_row = (lane_id / 4) + (mma_m * MMA_M); 
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
                int current_col = (lane_id % 4) * 2 + (mma_n * MMA_N); 
                #pragma unroll
                for(int i = 0; i < 2; i++){
                    streaming_store_f32x2(
                        &blockC[(current_row + i * 8) * N + current_col], 
                        make_float2(rC[mma_m][mma_n][2*i], rC[mma_m][mma_n][2*i + 1])
                    );
                }
            }
        }
    };

    int N_BLOCKS = N / BLOCK_N;
    int M_BLOCKS = M / BLOCK_M;
    int NUM_BLOCKS = M_BLOCKS * N_BLOCKS;
    int bid = blockIdx.x;
    int stage_id = 0;
    gC += (wid_x * WARP_M) * N + (wid_y * WARP_N);
    if (warp_id < 8){
        while(bid < NUM_BLOCKS){
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            for(int iter_k = 0; iter_k < K / BLOCK_K; iter_k ++){
                mbarrier_wait(load_mbar + stage_id * 8, (load_phases >> stage_id) & 1);
                load_phases ^= (1 << stage_id);
                load_regs(0, stage_id, 0);
                #pragma unroll
                for(int mma_k = 1; mma_k < BLOCK_K / MMA_K; mma_k++){
                    load_regs(mma_k, stage_id, mma_k & 1);
                    compute((mma_k - 1) & 1, (iter_k != 0) || (mma_k != 1));
                }
                mbarrier_arrive(compute_mbar + stage_id * 8);
                compute((BLOCK_K / MMA_K - 1) & 1, true);
                stage_id = (stage_id + 1) % NUM_STAGES;
            }
            store(gC + off_m * N + off_n);
            bid += gridDim.x;
        }
    }
    else if (warp_id == 8){
        while(bid < NUM_BLOCKS){
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            for(int iter_k = 0; iter_k < K / BLOCK_K; iter_k ++){
                mbarrier_wait(compute_mbar + stage_id * 8, (compute_phases >> stage_id) & 1);
                compute_phases ^= (1 << stage_id);
                if (elect_sync()){
                    mbarrier_arrive_expect_tx(load_mbar + stage_id * 8, STAGE_SIZE);
                    load_smem(iter_k, stage_id, off_n, off_m);
                }
                stage_id = (stage_id + 1) % NUM_STAGES;
            }
            bid += gridDim.x;
        }
    }
}

static CUtensorMap make_tma_descriptor_fp4_A(void *global_addr, uint64_t M,
                                             uint32_t K, uint32_t BLOCK_M,
                                             uint32_t BLOCK_K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K, M};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {BLOCK_K, BLOCK_M};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

static CUtensorMap make_tma_descriptor_fp4_B(void *global_addr, uint64_t N,
                                             uint64_t K, uint32_t BLOCK_N,
                                             uint32_t BLOCK_K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K, N};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {BLOCK_K, BLOCK_N};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

torch::Tensor nvfp4_gemm_v7(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb){
    int M = a.size(0);
    int N = b.size(0);
    int K = a.size(1) * 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor c = torch::empty({M, N}, options);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 2;
    dim3 grid(70, 1, 1);
    dim3 block(9 * 32, 1, 1);
    CUtensorMap tma_A = make_tma_descriptor_fp4_A(a.view(torch::kUInt8).data_ptr<uint8_t>(), M, K, BLOCK_M, BLOCK_K);
    CUtensorMap tma_B = make_tma_descriptor_fp4_B(b.view(torch::kUInt8).data_ptr<uint8_t>(), N, K, BLOCK_N, BLOCK_K);
    _nvfp4_gemm_v7<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES><<<grid, block>>>(sfa.view(torch::kUInt8).data_ptr<uint8_t>(), sfb.view(torch::kUInt8).data_ptr<uint8_t>(), tma_A, tma_B, c.data_ptr<float>(), M, N, K);
    return c;
}