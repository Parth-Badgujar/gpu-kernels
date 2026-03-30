#include "../../include/fence.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/others.cuh"
#include "../../include/tcgen05_mma.cuh"
#include "../../include/tcgen05_mov.cuh"
#include "../../include/tma.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

using fp16 = __half;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_STAGES>
__global__ void _matmul_nvfp4_v2(const __grid_constant__ CUtensorMap tmap_A,
                                 const __grid_constant__ CUtensorMap tmap_B,
                                 const uint8_t *d_a_scale,
                                 const uint8_t *d_b_scale, fp16 *d_c_mat, int M,
                                 int N, int K) {
    constexpr int SFA_SIZE = BLOCK_M * BLOCK_K / 16;
    constexpr int SFB_SIZE = BLOCK_N * BLOCK_K / 16;
    constexpr int A_SIZE = BLOCK_M * BLOCK_K / 2;
    constexpr int B_SIZE = BLOCK_N * BLOCK_K / 2;
    constexpr int STAGE_SIZE = A_SIZE + B_SIZE + SFA_SIZE + SFB_SIZE;
    constexpr int MMA_K = 64;
    const int n_iters = K / BLOCK_K;
    __shared__ uint8_t smem[STAGE_SIZE * NUM_STAGES];
    uint32_t smem_ptr = __cvta_generic_to_shared(smem);
    uint32_t A_shared = smem_ptr;
    uint32_t B_shared = A_shared + A_SIZE * NUM_STAGES;
    uint32_t sfa_shared = B_shared + B_SIZE * NUM_STAGES;
    uint32_t sfb_shared = sfa_shared + SFA_SIZE * NUM_STAGES;
    __shared__ uint64_t mbar_tma[NUM_STAGES];
    __shared__ uint64_t mbar_mm[NUM_STAGES];
    __shared__ uint32_t tmem[1];
    __shared__ int tma_phase[NUM_STAGES];
    __shared__ int mm_phase[NUM_STAGES];
    if (threadIdx.x < NUM_STAGES) {
        tma_phase[threadIdx.x] = 0;
        mm_phase[threadIdx.x] = 0;
    }
    uint32_t mbar_tma_ptr = __cvta_generic_to_shared(mbar_tma);
    uint32_t mbar_mm_ptr = __cvta_generic_to_shared(mbar_mm);
    uint32_t warp_id = threadIdx.x / 32;
    fp16 *d_c_mat_block =
        d_c_mat + BLOCK_M * blockIdx.x * N + BLOCK_N * blockIdx.y;

    if (warp_id == 0)
        tcgen_alloc(tmem, 512);
    if (warp_id == 1 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_init(mbar_tma_ptr + i * 8, 1);
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_init(mbar_mm_ptr + i * 8, 1);
        fence_proxy_async();
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_arrive(mbar_mm_ptr + i * 8);
    }
    __syncthreads();

    auto load_a_tma = [&](int i, int stage_id) {
        int off_m = blockIdx.x * BLOCK_M;
        int off_k = (BLOCK_K / 2) * i;
        uint32_t smem_a = A_shared + A_SIZE * stage_id;
        tma_load_2d(smem_a, (void *)&tmap_A, mbar_tma_ptr + stage_id * 8, off_k,
                    off_m);
    };
    auto load_b_tma = [&](int i, int stage_id) {
        int off_n = blockIdx.y * BLOCK_N;
        int off_k = (BLOCK_K / 2) * i;
        uint32_t smem_b = B_shared + B_SIZE * stage_id;
        tma_load_2d(smem_b, (void *)&tmap_B, mbar_tma_ptr + stage_id * 8, off_k,
                    off_n);
    };
    auto load_sfa_tma = [&](int i, int stage_id) {
        int off_m = blockIdx.x * (BLOCK_M * K / 16);
        int off_k = (128 * 4) * (BLOCK_K / MMA_K) * i;
        uint32_t smem_sfa = sfa_shared + SFA_SIZE * stage_id;
        tma_load_flat(smem_sfa, d_a_scale + off_m + off_k, SFA_SIZE,
                      mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };
    auto load_sfb_tma = [&](int i, int stage_id) {
        int off_n = blockIdx.y * (BLOCK_N * K / 16);
        int off_k = (128 * 4) * (BLOCK_K / MMA_K) * i;
        uint32_t smem_sfb = sfb_shared + SFB_SIZE * stage_id;
        tma_load_flat(smem_sfb, d_b_scale + off_n + off_k, SFB_SIZE / 2,
                      mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        tma_load_flat(smem_sfb + SFB_SIZE / 2,
                      d_b_scale + off_n + (BLOCK_N * K / 16) / 2 + off_k,
                      SFB_SIZE / 2, mbar_tma_ptr + stage_id * 8,
                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };
    uint64_t idesc = make_idesc_mxf4nvf4(128, 256);

    auto compute = [&](int i, int stage_id) {
        for (int mma_k = 0; mma_k < BLOCK_K / MMA_K; mma_k++) {
            uint64_t sfa_desc = make_smem_descriptor(
                sfa_shared + stage_id * SFA_SIZE + mma_k * 512, 8 * 16);
            uint64_t sfb1_desc = make_smem_descriptor(
                sfb_shared + stage_id * SFB_SIZE + mma_k * 512, 8 * 16);
            uint64_t sfb2_desc = make_smem_descriptor(
                sfb_shared + stage_id * SFB_SIZE + mma_k * 512 + SFB_SIZE / 2,
                8 * 16);
            uint64_t a_desc = make_smem_descriptor(
                A_shared + stage_id * A_SIZE + mma_k * 32, 8 * 128, 16, 2);
            uint64_t b_desc = make_smem_descriptor(
                B_shared + stage_id * B_SIZE + mma_k * 32, 8 * 128, 16, 2);
            tcgen_cp_32x128_warpx4(tmem[0], sfa_desc);
            tcgen_cp_32x128_warpx4(tmem[0] + 4, sfb1_desc);
            tcgen_cp_32x128_warpx4(tmem[0] + 8, sfb2_desc);
            tcgen05_mma_mxf4nvf4_4x(tmem[0] + 12, a_desc, b_desc, idesc,
                                    tmem[0], tmem[0] + 4, i | mma_k);
        }
        tcgen_commit_arrive_one(mbar_mm_ptr + stage_id * 8);
    };
    auto store = [&]() {
        for (int i = 0; i < BLOCK_N / 128; i++) {
            float tmp[128];
            int row = warp_id * 32;
            int col = i * 128;
            const int addr = (tmem[0] + 12) + (row << 16) + col;
            tcgen_ld_32x32_x128(tmp, addr);
            tcgen_ld_wait_sync();
            __half2 out[64];
            for (int j = 0; j < 64; j++)
                out[j] = __float22half2_rn({tmp[j * 2], tmp[j * 2 + 1]});
            for (int j = 0; j < 8; j++)
                store256(d_c_mat_block + i * 128 + j * 16 + threadIdx.x * N,
                         (uint32_t *)(&out[8 * j]));
        }
    };

    // Prologue
    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < min(NUM_STAGES - 1, n_iters); i++) {
            mbarrier_wait(mbar_mm_ptr + i * 8, mm_phase[i]);
            mm_phase[i] ^= 1;
            mbarrier_arrive_expect_tx(mbar_tma_ptr + i * 8, STAGE_SIZE);
            load_a_tma(i, i);
            load_b_tma(i, i);
            load_sfa_tma(i, i);
            load_sfb_tma(i, i);
        }
    }
    // Mainloop
    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < n_iters; i++) {
            int compute_iter = i;
            int curr_stage_id = compute_iter % NUM_STAGES;
            int prefetch_iter = compute_iter + NUM_STAGES - 1;

            if (prefetch_iter < n_iters) {
                int next_stage_id = prefetch_iter % NUM_STAGES;

                mbarrier_wait(mbar_mm_ptr + next_stage_id * 8,
                              mm_phase[next_stage_id]);
                mm_phase[next_stage_id] ^= 1;

                mbarrier_arrive_expect_tx(mbar_tma_ptr + next_stage_id * 8,
                                          STAGE_SIZE);
                load_a_tma(prefetch_iter, next_stage_id);
                load_b_tma(prefetch_iter, next_stage_id);
                load_sfa_tma(prefetch_iter, next_stage_id);
                load_sfb_tma(prefetch_iter, next_stage_id);
            }
            mbarrier_wait(mbar_tma_ptr + curr_stage_id * 8,
                          tma_phase[curr_stage_id]);
            tma_phase[curr_stage_id] ^= 1;
            compute(compute_iter, curr_stage_id);
        }
    }

    __syncthreads();
    tcgen_after_thread_sync();
    store();
    if (warp_id == 0)
        tcgen_delloc(tmem[0], 512);
}

static CUtensorMap make_tma_descriptor_fp4_A(void *global_addr, uint64_t M,
                                             uint64_t K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K / 2, M};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {128, 128};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

static CUtensorMap make_tma_descriptor_fp4_B(void *global_addr, uint64_t N,
                                             uint64_t K) {
    CUtensorMap tma_desc;
    uint64_t globalDim[2] = {K / 2, N};
    uint64_t globalStrides[1] = {K / 2};
    uint32_t boxDim[2] = {128, 256};
    uint32_t elementStrides[2] = {1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

void matmul_nvfp4_v2(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 256;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 4;
    dim3 grid(M / BLOCK_M, N / BLOCK_N);
    dim3 block(32 * 4);

    CUtensorMap tmap_A = make_tma_descriptor_fp4_A((void *)a, M, K);
    CUtensorMap tmap_B = make_tma_descriptor_fp4_B((void *)b, N, K);

    _matmul_nvfp4_v2<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>
        <<<grid, block>>>(tmap_A, tmap_B, sfa, sfb, c, M, N, K);
}
