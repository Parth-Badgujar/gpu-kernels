#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include "../../include/mma_sync.cuh"
#include "../../include/tma.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/fence.cuh"



#define MMA_K 64

__device__ __forceinline__ uint32_t swizzle_ptr(uint32_t ptr){
    uint32_t y = (ptr / 128) % 8;
    uint32_t x = (ptr / 16) % 8;
    uint32_t new_x = x ^ y;
    return (ptr & (~(0x7 << 4))) | (new_x << 4);
}


template<int BLOCK_M = 32, int BLOCK_N = 32, int MMA_M, int MMA_N, int NUM_STAGES = 2>
__global__ void _nvfp4_gemm_v1(
    const uint8_t* __restrict__ gSFA,
    const uint8_t* __restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    float* __restrict__ c,
    int M,
    int N,
    int K
)
{
    constexpr int sA_size = (BLOCK_M * MMA_K / 2);
    constexpr int sB_size = (BLOCK_N * MMA_K / 2);
    constexpr int s_sfa_size = (BLOCK_M * MMA_K / 16);
    constexpr int s_sfb_size = (BLOCK_N * MMA_K / 16);
    constexpr int STAGE_SIZE = sA_size + sB_size + s_sfa_size + s_sfb_size;
    __shared__ uint8_t smem[NUM_STAGES * STAGE_SIZE];
    uint32_t smem_base = __cvta_generic_to_shared(smem);
    uint32_t sA    = smem_base;
    uint32_t sB    = sA + sA_size;
    uint32_t s_sfa = sB + sB_size;
    uint32_t s_sfb = s_sfa + s_sfa_size;
    uint32_t* sfa = (uint32_t*)__cvta_shared_to_generic(s_sfa);
    uint32_t* sfb = (uint32_t*)__cvta_shared_to_generic(s_sfb);
    static_assert(MMA_M % 16 == 0);
    static_assert(MMA_N % 8 == 0);
    static_assert(BLOCK_M % MMA_M == 0);
    static_assert(BLOCK_N % MMA_N == 0);
    static_assert((MMA_M / 16) * (MMA_N / 8) * 32 == blockDim.x * blockDim.y);
    int warp_id = threadIdx.x / 32;
    int wid_x = warp_id / 4;
    int wid_y = warp_id % 4;
    int lane_id = threadIdx.x % 32;
    int quad_id = threadIdx.x % 4;
    uint32_t rA[BLOCK_M / MMA_M][MMA_K / 16][4];
    uint32_t rB[BLOCK_N / MMA_N][MMA_K / 16][2];
    uint32_t r_sfa[BLOCK_M / MMA_M];
    uint32_t r_sfb[BLOCK_N / MMA_N];
    float rC[BLOCK_M / MMA_M][BLOCK_N / MMA_N][4];


    for(int iter_k = 0; iter_k < K / MMA_K; iter_k++){

        tma_load_2d(sA, &tmap_A, mbar, blockIdx.x * BLOCK_M, blockIdx.y * MMA_K);
        tma_load_2d(sB, &tmap_B, mbar, blockIdx.y * BLOCK_N, blockIdx.x * MMA_K);

        for(int mma_m = 0; mma_m < BLOCK_M / MMA_M; mma_m++){
            int off_m = (mma_m * MMA_M + wid_x + (threadIdx.x % 16)) * (MMA_K / 2);
            int off_k = wid_y * 32 + (threadIdx.x / 16) * 16;
            ldmatrix_m16n16_x2_b8(rA[mma_m][iter_k], sA + off_m + off_k);
            r_sfa[mma_m] = sfa[(7 - (threadIdx.x % 8)) + (quad_id & 1) * 8 + mma_m * 16];
        }
        for(int mma_n = 0; mma_n < BLOCK_N / MMA_N; mma_n++){
            int off_n = (mma_n * MMA_N + wid_x + (threadIdx.x % 8)) * (MMA_K / 2);
            int off_k = wid_y * 32 + (threadIdx.x / 8) * 16;
            ldmatrix_m8n8_x2_b16(rB[mma_n][iter_k], sB + off_n + off_k);
            r_sfb[mma_n] = sfb[mma_n * 8 + (7 - (threadIdx.x % 8))];
        }

        for(int mma_m = 0; mma_m < BLOCK_M / MMA_M; mma_m++){
            for(int mma_n = 0; mma_n < BLOCK_N / MMA_N; mma_n++){
                mma_m16n8k64_mxf4nvf4_4x_ue4m3(
                    rC[mma_m][mma_n],
                    rA[mma_m][iter_k],
                    rB[mma_n][iter_k],
                    rC[mma_m][mma_n],
                    r_sfa[mma_m],
                    r_sfb[mma_n]
                );
            }
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
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    // CUtensorMap tma_desc;
    // constexpr int SWIZZLE_SIZE = 128;
    // uint64_t globalDim[3] = {SWIZZLE_SIZE, M, (K / 2) / SWIZZLE_SIZE};
    // uint64_t globalStrides[2] = {K / 2, SWIZZLE_SIZE};
    // uint32_t boxDim[3] = {SWIZZLE_SIZE, BLOCK_M, (BLOCK_K / 2) /
    // SWIZZLE_SIZE}; uint32_t elementStrides[3] = {1, 1, 1};
    // cuTensorMapEncodeTiled(
    //     &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, global_addr, globalDim,
    //     globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    // CUtensorMap tma_desc;
    // constexpr int SWIZZLE_SIZE = 128;
    // uint64_t globalDim[3] = {SWIZZLE_SIZE, N, (K / 2) / SWIZZLE_SIZE};
    // uint64_t globalStrides[2] = {K / 2, SWIZZLE_SIZE};
    // uint32_t boxDim[3] = {SWIZZLE_SIZE, BLOCK_N, (BLOCK_K / 2) /
    // SWIZZLE_SIZE}; uint32_t elementStrides[3] = {1, 1, 1};
    // cuTensorMapEncodeTiled(
    //     &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, global_addr, globalDim,
    //     globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
    //     CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
    //     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}