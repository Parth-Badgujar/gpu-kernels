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

template<int BLOCK_M = 128, int BLOCK_N = 128, int MMA_M = 32, int MMA_N = 32, int NUM_STAGES = 1>
__global__ void _nvfp4_gemm_v1(
    const uint8_t* __restrict__ gSFA,
    const uint8_t* __restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    float* __restrict__ gC,
    int M, int N, int K
)
{
    constexpr int sA_mem   = (BLOCK_M * MMA_K / 2);
    constexpr int sB_mem   = (BLOCK_N * MMA_K / 2);
    constexpr int sSFA_mem = (BLOCK_M * MMA_K / 16);
    constexpr int sSFB_mem = (BLOCK_N * MMA_K / 16);
    constexpr int STAGE_SIZE = sA_mem + sB_mem + sSFA_mem + sSFB_mem;

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int quad_id = lane_id / 4;
    int lane_id_in_quad = lane_id % 4;
    int wid_x   = warp_id / 4;
    int wid_y   = warp_id % 4;

    __shared__ uint8_t  smem[STAGE_SIZE];
    __shared__ uint64_t mbar[1];
    __shared__ int      phase;

    uint32_t smem_base = __cvta_generic_to_shared(smem);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);

    if (warp_id == 0 && elect_sync()) {
        phase = 0;
        mbarrier_init(mbar_addr, 1);
        fence_mbarrier_init();
    }
    __syncthreads();

    uint32_t sA   = smem_base;
    uint32_t sB   = sA   + sA_mem;
    uint32_t sSFA = sB   + sB_mem;
    uint32_t sSFB = sSFA + sSFA_mem;

    uint32_t* sfa_ptr = (uint32_t*)__cvta_shared_to_generic(sSFA);
    uint32_t* sfb_ptr = (uint32_t*)__cvta_shared_to_generic(sSFB);

    constexpr int TILES_M = BLOCK_M / MMA_M;
    constexpr int TILES_N = BLOCK_N / MMA_N;

    float    rC[TILES_M][TILES_N][4] = {};

    uint32_t rA[TILES_M][4];
    uint32_t rB[TILES_N][2];
    uint32_t rSFA[TILES_M];
    uint32_t rSFB[TILES_N];

    int off_m = blockIdx.x * BLOCK_M;
    int off_n = blockIdx.y * BLOCK_N;

    for (int iter_k = 0; iter_k < K / MMA_K; iter_k++) {
        if (warp_id == 0 && elect_sync()) {
            mbarrier_arrive_expect_tx(mbar_addr, STAGE_SIZE);
            tma_load_2d(sA,   &tmap_A, mbar_addr, iter_k * MMA_K, off_m);
            tma_load_2d(sB,   &tmap_B, mbar_addr, iter_k * MMA_K, off_n);
            tma_load_flat(sSFA, gSFA + off_m * (K / 16) + iter_k * (BLOCK_M * MMA_K / 16), sSFA_mem, mbar_addr, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
            tma_load_flat(sSFB, gSFB + off_n * (K / 16) + iter_k * (BLOCK_N * MMA_K / 16), sSFB_mem, mbar_addr, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        }
        mbarrier_wait(mbar_addr, phase);

        if (warp_id == 0 && elect_sync()) {
            phase ^= 1;
        }
        __syncthreads();

        for (int mma_m = 0; mma_m < TILES_M; mma_m++) {
            uint32_t row  = mma_m * MMA_M + wid_x * 16 + (lane_id % 16);
            uint32_t addr = sA + row * (MMA_K / 2) + (lane_id / 16) * 16;
            ldmatrix_m8n8_x4_b16(rA[mma_m], addr);
        }

        for (int mma_n = 0; mma_n < TILES_N; mma_n++) {
            uint32_t row  = mma_n * MMA_N + wid_y * 8 + (lane_id % 8);
            uint32_t addr = sB + row * (MMA_K / 2) + (lane_id / 8) * 16;
            ldmatrix_m8n8_x2_b16(rB[mma_n], addr);
        }

        for (int mma_m = 0; mma_m < TILES_M; mma_m += 2) {
            rSFA[mma_m / 2] = sfa_ptr[(warp_id / 4) * 64 + ((quad_id) + (lane_id_in_quad & 1) * 8) * 4 + (lane_id_in_quad >> 1) + mma_m];
        }

        for (int mma_n = 0; mma_n < TILES_N; mma_n += 4) {
            rSFB[mma_n / 4] = sfb_ptr[(warp_id % 4) * 32 + (quad_id) * 4 + lane_id_in_quad];
        }

        for (int mma_m = 0; mma_m < TILES_M; mma_m++) {
            for (int mma_n = 0; mma_n < TILES_N; mma_n++) {
                mma_m16n8k64_mxf4nvf4_4x_ue4m3(
                    rC[mma_m][mma_n],
                    rA[mma_m],
                    rB[mma_n],
                    rC[mma_m][mma_n],
                    rSFA[mma_m / 2],
                    rSFB[mma_n / 4],
                    0,
                    mma_m % 2,
                    0,
                    mma_n % 4
                );
            }
        }
        __syncthreads();
    }

    float* block_C = gC + off_m * N + off_n;
    for (int mma_m = 0; mma_m < TILES_M; mma_m++) {
        for (int mma_n = 0; mma_n < TILES_N; mma_n++) {
            for (int i = 0; i < 2; i++) {
                int x = mma_m * MMA_M + wid_x * 16 + (lane_id / 4) + i * 8;
                int y = mma_n * MMA_N + wid_y * 8  + (lane_id % 4) * 2;
                reinterpret_cast<int2*>(&block_C[x * N + y])[0] =
                    reinterpret_cast<int2*>(rC[mma_m][mma_n])[i];
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
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
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
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}


torch::Tensor nvfp4_gemm_v1(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb){
    int M = a.size(0);
    int N = b.size(0);
    int K = a.size(1) * 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor c = torch::empty({M, N}, options);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int NUM_STAGES = 1;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(256, 1, 1);
    CUtensorMap tma_A = make_tma_descriptor_fp4_A(a.view(torch::kUInt8).data_ptr<uint8_t>(), M, K, BLOCK_M, MMA_K);
    CUtensorMap tma_B = make_tma_descriptor_fp4_B(b.view(torch::kUInt8).data_ptr<uint8_t>(), N, K, BLOCK_N, MMA_K);
    _nvfp4_gemm_v1<BLOCK_M, BLOCK_N, 32, 32, NUM_STAGES><<<grid, block>>>(sfa.view(torch::kUInt8).data_ptr<uint8_t>(), sfb.view(torch::kUInt8).data_ptr<uint8_t>(), tma_A, tma_B, c.data_ptr<float>(), M, N, K);
    return c;
}