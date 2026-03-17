#include "../../blackwell_helpers/fence.cuh"
#include "../../blackwell_helpers/mbarrier.cuh"
#include "../../blackwell_helpers/others.cuh"
#include "../../blackwell_helpers/tcgen05_mma.cuh"
#include "../../blackwell_helpers/tcgen05_mov.cuh"
#include "../../blackwell_helpers/tma.cuh"
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using fp8 = __nv_fp8_e4m3;
using fp16 = __half;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void _matmul_nvfp4_v1(const __grid_constant__ CUtensorMap tmap_A,
                                 const __grid_constant__ CUtensorMap tmap_B,
                                 const __grid_constant__ CUtensorMap tmap_sfa,
                                 const __grid_constant__ CUtensorMap tmap_sfb,
                                 int stride_cm, int stride_cn, fp16 *d_c_mat) {
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    fp16 *c_mat =
        d_c_mat + stride_cn * BLOCK_N * block_y + stride_cm * BLOCK_M * block_x;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint8_t smem[BLOCK_M * BLOCK_K / 2 + BLOCK_N * BLOCK_K / 2 +
                            BLOCK_M * BLOCK_K / 16 + BLOCK_N * BLOCK_K / 16];
    uint32_t smem_ptr = __cvta_generic_to_shared(smem);
    uint32_t a_shared = smem_ptr;
    uint32_t b_shared = a_shared + BLOCK_M * BLOCK_K / 2;
    uint32_t sfa_shared = b_shared + BLOCK_N * BLOCK_K / 2;
    uint32_t sfb_shared = sfa_shared + BLOCK_M * BLOCK_K / 16;
    uint32_t tmem_sfa = 0;
    uint32_t tmem_sfb = tmem_sfa + 4;
    uint32_t tmem_d = tmem_sfb + 8;
    __shared__ uint64_t mbarrier[1];
    __shared__ uint32_t tmem_addr[1];
    uint32_t mbar_ptr = __cvta_generic_to_shared(mbarrier);
    const int warp_id = threadIdx.x / 32;

    uint32_t idesc = make_idesc_mxf4nvf4(BLOCK_M, BLOCK_N);
    if (warp_id == 0 && elect_sync()) {
        mbarrier_init(mbar_ptr, 1);
        fence_proxy_async();
    }
    __syncthreads();
    if (warp_id == 0 && elect_sync()) {
        mbarrier_arrive_expect_tx(
            mbar_ptr, BLOCK_M * BLOCK_K / 2 + BLOCK_N * BLOCK_K / 2 +
                          BLOCK_M * BLOCK_K / 16 + BLOCK_N * BLOCK_K / 16);
        for (int i = 0; i < BLOCK_K / 32; i++) {
            tma_load_2d(a_shared + 16 * BLOCK_M * i, &tmap_A, mbar_ptr, 16 * i,
                        BLOCK_M * block_x);
            tma_load_2d(b_shared + 16 * BLOCK_N * i, &tmap_B, mbar_ptr, 16 * i,
                        BLOCK_N * block_y);
        }
        tma_load_4d(sfa_shared, &tmap_sfa, mbar_ptr, 0, 0, 0, block_x);
        tma_load_4d(sfb_shared, &tmap_sfb, mbar_ptr, 0, 0, 0, block_y * 2);
        tma_load_4d(sfb_shared + 512, &tmap_sfb, mbar_ptr, 0, 0, 0,
                    block_y * 2 + 1);
    }
    __syncthreads();
    if (warp_id == 0) {
        tcgen_alloc(tmem_addr, 512);
    }
    uint64_t smem_desc_sfa = make_smem_descriptor(sfa_shared, 8 * 16, 0, 0);
    uint64_t smem_desc_sfb1 = make_smem_descriptor(sfb_shared, 8 * 16, 0, 0);
    uint64_t smem_desc_sfb2 =
        make_smem_descriptor(sfb_shared + 512, 8 * 16, 0, 0);
    uint64_t smem_desc_a =
        make_smem_descriptor(a_shared, 8 * 16, BLOCK_M * 16, 0);
    uint64_t smem_desc_b =
        make_smem_descriptor(b_shared, 8 * 16, BLOCK_N * 16, 0);

    mbarrier_wait(mbar_ptr, 0);
    if (warp_id == 0 && threadIdx.x == 0) {
        tcgen_cp_32x128_warpx4(tmem_sfa, smem_desc_sfa);
        tcgen_cp_32x128_warpx4(tmem_sfb, smem_desc_sfb1);
        tcgen_cp_32x128_warpx4(tmem_sfb + 4, smem_desc_sfb2);
        tcgen05_mma_mxf4nvf4_4x(tmem_d, smem_desc_a, smem_desc_b, idesc,
                                tmem_sfa, tmem_sfb, 0);
        tcgen_commit_arrive_one(mbar_ptr);
    }
    mbarrier_wait(mbar_ptr, 1);
    tcgen_after_thread_sync();
    for (int i = 0; i < BLOCK_N / 8; i++) {
        float tmp[8];
        int row = warp_id * 32;
        int col = i * 8;
        const int addr = tmem_d + (row << 16) + col;
        tcgen_ld_32x32_x8(tmp, addr);
        tcgen_ld_wait_sync();
        __half2 out[4];
        for (int j = 0; j < 4; j++)
            out[j] = __float22half2_rn({tmp[j * 2], tmp[j * 2 + 1]});
        reinterpret_cast<int4 *>(c_mat + i * 8 + threadIdx.x * stride_cm)[0] =
            reinterpret_cast<int4 *>(out)[0];
    }
    __syncthreads();
    if (warp_id == 0) {
        tcgen_delloc(0, 512);
    }
}

static CUtensorMap make_tma_descriptor_fp4_A(void *global_addr, uint64_t M,
                                             uint64_t K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K / 2, M};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {16, 128};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

static CUtensorMap make_tma_descriptor_fp4_B(void *global_addr, uint64_t N,
                                             uint64_t K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K / 2, N};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {16, 256};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

static CUtensorMap make_tma_descriptor_sf(void *global_addr, uint64_t M,
                                          uint64_t K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[4] = {128, 4, (K / 16) / 4, M / 128};
    uint64_t globalStrides[3] = {128, 128 * 4, 128 * 4 * (K / 16) / 4};
    uint32_t boxDim[4] = {128, 4, 1, 1};
    uint32_t elementStrides[4] = {1, 1, 1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

void matmul_nvfp4_v1(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K) {
    assert(K == 64);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_K = 64;
    dim3 grid(M / BLOCK_M, N / BLOCK_N);
    dim3 block(32 * 4);

    CUtensorMap tmap_A = make_tma_descriptor_fp4_A((void *)a, K, M);
    CUtensorMap tmap_B = make_tma_descriptor_fp4_B((void *)b, N, K);
    CUtensorMap tmap_sfA = make_tma_descriptor_sf((void *)sfa, M, K);
    CUtensorMap tmap_sfB = make_tma_descriptor_sf((void *)sfb, N, K);

    int stride_cm = N;
    int stride_cn = 1;
    _matmul_nvfp4_v1<BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(
        tmap_A, tmap_B, tmap_sfA, tmap_sfB, stride_cm, stride_cn, c);
    cudaDeviceSynchronize();
}