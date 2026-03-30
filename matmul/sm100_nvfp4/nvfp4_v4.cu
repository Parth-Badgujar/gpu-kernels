#include "../../include/clc.cuh"
#include "../../include/fence.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/others.cuh"
#include "../../include/tcgen05_mma.cuh"
#include "../../include/tcgen05_mov.cuh"
#include "../../include/tma.cuh"
#include "profiler.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda/ptx>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define MMA_K 64
#define MMA_M 128
#define MMA_N 256

using fp8 = __nv_fp8_e4m3;
using fp16 = __half;

template <int BLOCK_M = 128, int BLOCK_N = 256, int BLOCK_K, int NUM_STAGES>
__global__ void _matmul_nvfp4_v4(const __grid_constant__ CUtensorMap tmap_A,
                                 const __grid_constant__ CUtensorMap tmap_B,
                                 const uint8_t *sfa, const uint8_t *sfb,
                                 fp16 *d_c_mat, int M, int N, int K) {
    constexpr int SFA_SIZE = BLOCK_M * BLOCK_K / 16;
    constexpr int SFB_SIZE = BLOCK_N * BLOCK_K / 16;
    constexpr int A_SIZE = BLOCK_M * BLOCK_K / 2;
    constexpr int B_SIZE = BLOCK_N * BLOCK_K / 2;
    constexpr int STAGE_SIZE = A_SIZE + B_SIZE + SFA_SIZE + SFB_SIZE;
    const int n_iters = K / BLOCK_K;
    __shared__ uint8_t smem[STAGE_SIZE * NUM_STAGES];
    uint32_t smem_ptr = __cvta_generic_to_shared(smem);
    uint32_t A_shared = smem_ptr;
    uint32_t B_shared = A_shared + A_SIZE * NUM_STAGES;
    uint32_t sfa_shared = B_shared + B_SIZE * NUM_STAGES;
    uint32_t sfb_shared = sfa_shared + SFA_SIZE * NUM_STAGES;
    __shared__ uint64_t mbar_tma[NUM_STAGES];
    __shared__ uint64_t mbar_mm[NUM_STAGES];
    __shared__ uint64_t mbar_compute[1];
    __shared__ uint64_t clc_trigger[4];
    __shared__ uint32_t tmem[1];
    __shared__ int tma_phase[NUM_STAGES];
    __shared__ int mm_phase[NUM_STAGES];
    __shared__ int clc_trigger_phase, store_done_trigger_phase, clc_bar_phase,
        safe_store_trigger_phase, compute_phase;
    __shared__ int block_x, block_y, work_is_there;
    __shared__ uint4 clc_result[1];
    __syncthreads();
    uint32_t mbar_tma_ptr = __cvta_generic_to_shared(mbar_tma);
    uint32_t mbar_mm_ptr = __cvta_generic_to_shared(mbar_mm);
    uint32_t mbar_compute_ptr = __cvta_generic_to_shared(mbar_compute);
    uint32_t clc_trigger_ptr = __cvta_generic_to_shared(clc_trigger);
    uint32_t store_done_trigger_ptr = clc_trigger_ptr + 8;
    uint32_t safe_store_trigger_ptr = clc_trigger_ptr + 16;
    uint32_t clc_bar_ptr = clc_trigger_ptr + 24;
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t clc_result_ptr = __cvta_generic_to_shared(clc_result);

    if (threadIdx.x < NUM_STAGES) {
        tma_phase[threadIdx.x] = 0;
        mm_phase[threadIdx.x] = 0;
    }

    if (warp_id == 0 && elect_sync()) {
        uint64_t dry_run = globaltimer();
        dry_run = globaltimer() - dry_run;
        work_is_there = 1;
        compute_phase = 0;
        clc_trigger_phase = 0;
        store_done_trigger_phase = 0;
        safe_store_trigger_phase = 0;
        clc_bar_phase = 0;
        block_x = blockIdx.x;
        block_y = blockIdx.y;
    }

    if (warp_id == 4)
        tcgen_alloc(tmem, 512);
    if (warp_id == 1 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_init(mbar_tma_ptr + i * 8, 1);
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_init(mbar_mm_ptr + i * 8, 1);
        mbarrier_init(mbar_compute_ptr);
        mbarrier_init(clc_bar_ptr, 2);
        mbarrier_init(clc_trigger_ptr);
        mbarrier_init(store_done_trigger_ptr, 4);
        mbarrier_init(safe_store_trigger_ptr, 1);
        fence_proxy_async();
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_arrive(mbar_mm_ptr + i * 8);
        for (int i = 0; i < 4; i++)
            mbarrier_arrive(store_done_trigger_ptr);
    }
    __syncthreads();

    auto load_a_tma = [&](int i, int stage_id) {
        int off_m = block_x * BLOCK_M;
        uint32_t smem_a = A_shared + A_SIZE * stage_id;
        for (int k = 0; k < (BLOCK_K / 2) / 128; k++) {
            const int off_k = (BLOCK_K / 2) * i + k * 128;
            tma_load_2d(smem_a + k * BLOCK_M * 128, (void *)&tmap_A,
                        mbar_tma_ptr + stage_id * 8, off_k, off_m);
        }
    };
    auto load_b_tma = [&](int i, int stage_id) {
        int off_n = block_y * BLOCK_N;
        uint32_t smem_b = B_shared + B_SIZE * stage_id;
        for (int k = 0; k < (BLOCK_K / 2) / 128; k++) {
            const int off_k = (BLOCK_K / 2) * i + k * 128;
            tma_load_2d(smem_b + k * BLOCK_N * 128, (void *)&tmap_B,
                        mbar_tma_ptr + stage_id * 8, off_k, off_n);
        }
    };
    auto load_sfa_tma = [&](int i, int stage_id) {
        int off_m = block_x * (BLOCK_M * K / 16);
        int off_k = (128 * 4) * (BLOCK_K / MMA_K) * i;
        uint32_t smem_sfa = sfa_shared + SFA_SIZE * stage_id;
        tma_load_flat(smem_sfa, sfa + off_m + off_k, SFA_SIZE,
                      mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };
    auto load_sfb_tma = [&](int i, int stage_id) {
        int off_n = block_y * (BLOCK_N * K / 16);
        int off_k = (128 * 4) * (BLOCK_K / MMA_K) * i;
        uint32_t smem_sfb = sfb_shared + SFB_SIZE * stage_id;
        tma_load_flat(smem_sfb, sfb + off_n + off_k, SFB_SIZE / 2,
                      mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        tma_load_flat(smem_sfb + SFB_SIZE / 2,
                      sfb + off_n + (BLOCK_N * K / 16) / 2 + off_k,
                      SFB_SIZE / 2, mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };
    uint64_t idesc = make_idesc_mxf4nvf4(128, 256);

    auto compute = [&](int i, int stage_id) {
#pragma unroll
        for (int mma_k = 0; mma_k < BLOCK_K / MMA_K; mma_k++) {
            uint64_t sfa_desc = make_smem_descriptor(
                sfa_shared + stage_id * SFA_SIZE + mma_k * 512, 8 * 16);
            uint64_t sfb1_desc = make_smem_descriptor(
                sfb_shared + stage_id * SFB_SIZE + mma_k * 512, 8 * 16);
            uint64_t sfb2_desc = make_smem_descriptor(
                sfb_shared + stage_id * SFB_SIZE + mma_k * 512 + SFB_SIZE / 2,
                8 * 16);
            uint64_t a_desc = make_smem_descriptor(
                A_shared + stage_id * A_SIZE + (mma_k / 4) * (128 * BLOCK_M) +
                    (mma_k % 4) * 32,
                8 * 128, 16, 2);
            uint64_t b_desc = make_smem_descriptor(
                B_shared + stage_id * B_SIZE + (mma_k / 4) * (128 * BLOCK_N) +
                    (mma_k % 4) * 32,
                8 * 128, 16, 2);
            tcgen_cp_32x128_warpx4(tmem[0], sfa_desc);
            tcgen_cp_32x128_warpx4(tmem[0] + 4, sfb1_desc);
            tcgen_cp_32x128_warpx4(tmem[0] + 8, sfb2_desc);
            tcgen05_mma_mxf4nvf4_4x(tmem[0] + 12, a_desc, b_desc, idesc,
                                    tmem[0], tmem[0] + 4, i | mma_k);
        }
    };
    auto store = [&](fp16 *c_block) {
        int warp_mod = warp_id % 4;
#pragma unroll
        for (int i = 0; i < BLOCK_N / 128; i++) {
            float tmp[128];
            int row = warp_mod * 32;
            int col = i * 128;
            const int addr = (tmem[0] + 12) + (row << 16) + col;
            tcgen_ld_32x32_x128(tmp, addr);
            tcgen_ld_wait_sync();
            __half2 out[64];
            for (int j = 0; j < 64; j++)
                out[j] = __float22half2_rn({tmp[j * 2], tmp[j * 2 + 1]});
            for (int j = 0; j < 8; j++)
                store256(c_block + i * 128 + j * 16 + (threadIdx.x % 128) * N,
                         (uint32_t *)(&out[8 * j]));
        }
    };

    if (warp_id == 0 && elect_sync()) {
        int iter = 0;
        while (work_is_there) {
            for (int i = 0; i < n_iters; i++) {
                int stage_id = i % NUM_STAGES;
                mbarrier_wait(mbar_mm_ptr + stage_id * 8, mm_phase[stage_id]);
                mm_phase[stage_id] ^= 1;
                mbarrier_arrive_expect_tx(mbar_tma_ptr + stage_id * 8,
                                          STAGE_SIZE);
                load_a_tma(i, stage_id);
                load_b_tma(i, stage_id);
                load_sfa_tma(i, stage_id);
                load_sfb_tma(i, stage_id);
            }
            mbarrier_wait(clc_trigger_ptr, clc_trigger_phase);
            clc_trigger_phase ^= 1;
            iter++;
        }
    } else if (warp_id == 1 && elect_sync()) {
        int iter = 0;
        while (work_is_there) {
            mbarrier_wait(store_done_trigger_ptr, store_done_trigger_phase);
            store_done_trigger_phase ^= 1;
            for (int i = 0; i < n_iters; i++) {
                int stage_id = i % NUM_STAGES;
                mbarrier_wait(mbar_tma_ptr + stage_id * 8, tma_phase[stage_id]);
                tma_phase[stage_id] ^= 1;
                compute(i, stage_id);
                tcgen_commit_arrive_one(mbar_mm_ptr + stage_id * 8);
            }
            mbarrier_arrive(mbar_compute_ptr);
            mbarrier_wait(clc_trigger_ptr, clc_trigger_phase);
            iter++;
        }
    } else if (1 < warp_id && warp_id < 6) {
        int iter = 0;
        while (work_is_there) {
            mbarrier_wait(mbar_compute_ptr, compute_phase);
            if (warp_id == 4 && elect_sync())
                compute_phase ^= 1;
            tcgen_after_thread_sync();
            fp16 *c_block = d_c_mat + BLOCK_M * block_x * N + BLOCK_N * block_y;
            if (elect_sync() && warp_id == 4) {
                mbarrier_arrive(clc_bar_ptr);
            }
            store(c_block);
            iter++;
            if (elect_sync())
                mbarrier_arrive(store_done_trigger_ptr);
            mbarrier_wait(safe_store_trigger_ptr, safe_store_trigger_phase);
            if (warp_id == 4 && elect_sync())
                safe_store_trigger_phase ^= 1;
        }
        if (warp_id == 4)
            tcgen_delloc(tmem[0], 512);
    } else if (warp_id == 6 && elect_sync()) {
        while (work_is_there) {
            mbarrier_arrive_expect_tx(clc_bar_ptr, 16);
            clc_try_cancel(clc_result_ptr, clc_bar_ptr);
            mbarrier_wait(clc_bar_ptr, clc_bar_phase);
            clc_bar_phase ^= 1;
            if (clc_query_is_canceled(clc_result[0])) {
                int ctaid_x = clc_query_ctaid_x(clc_result[0]);
                int ctaid_y = clc_query_ctaid_y(clc_result[0]);
                int ctaid_z = clc_query_ctaid_z(clc_result[0]);
                block_x = ctaid_x;
                block_y = ctaid_y;
                mbarrier_arrive(clc_trigger_ptr);
                mbarrier_arrive(safe_store_trigger_ptr);
            } else {
                work_is_there = 0;
                mbarrier_arrive(safe_store_trigger_ptr);
                mbarrier_arrive(clc_trigger_ptr);
            }
        }
    }
}

static CUtensorMap make_tma_descriptor_fp4_A(void *global_addr, uint64_t M,
                                             uint32_t K, uint32_t BLOCK_M,
                                             uint32_t BLOCK_K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K / 2, M};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {128, BLOCK_M};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
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
    uint64_t globalDim[2] = {K / 2, N};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {128, BLOCK_N};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
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

void matmul_nvfp4_v4(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 256;
    dim3 grid(M / BLOCK_M, N / BLOCK_N);
    dim3 block(32 * 7);
    if (K == 256) {
        CUtensorMap tmap_A =
            make_tma_descriptor_fp4_A((void *)a, M, K, BLOCK_M, 256);
        CUtensorMap tmap_B =
            make_tma_descriptor_fp4_B((void *)b, N, K, BLOCK_N, 256);
        _matmul_nvfp4_v4<BLOCK_M, BLOCK_N, 256, 4>
            <<<grid, block>>>(tmap_A, tmap_B, sfa, sfb, c, M, N, K);
    } else {
        CUtensorMap tmap_A =
            make_tma_descriptor_fp4_A((void *)a, M, K, BLOCK_M, 512);
        CUtensorMap tmap_B =
            make_tma_descriptor_fp4_B((void *)b, N, K, BLOCK_N, 512);
        _matmul_nvfp4_v4<BLOCK_M, BLOCK_N, 512, 2>
            <<<grid, block>>>(tmap_A, tmap_B, sfa, sfb, c, M, N, K);
    }
}