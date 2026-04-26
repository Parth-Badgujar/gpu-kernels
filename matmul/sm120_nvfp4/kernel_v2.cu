#include "../../include/ldmatrix.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/mma_sync.cuh"
#include "../../include/others.cuh"
#include "../../include/tma.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#define MMA_K 64
#define MMA_M 16
#define MMA_N 8

__device__ __forceinline__
int phase_idx(uint32_t &phase, int idx) {
    return (phase >> idx) & 1;
}

__device__ __forceinline__
void phase_flip_idx(uint32_t &phase, int idx) {
    phase ^= (1 << idx);
}

template <int BLOCK_M = 128, int BLOCK_N = 128, int BLOCK_K = 64, int NUM_STAGES = 1>
__global__ void kernel_v2(
    const uint8_t *__restrict__ gSFA,
    const uint8_t *__restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    float *__restrict__ gC,
    int M, int N, int K
) {
    constexpr int sA_mem = (BLOCK_M * BLOCK_K / 2);
    constexpr int sB_mem = (BLOCK_N * BLOCK_K / 2);
    constexpr int sSFA_mem = (BLOCK_M * BLOCK_K / 16);
    constexpr int sSFB_mem = (BLOCK_N * BLOCK_K / 16);
    constexpr int STAGE_SIZE = sA_mem + sB_mem + sSFA_mem + sSFB_mem;
    // 128x128 -> 64x64 (WARP_M = WARP_N = 64)
    constexpr int WARP_M = BLOCK_M / 2;
    constexpr int WARP_N = BLOCK_N / 2;

    __shared__ uint8_t smem_generic[STAGE_SIZE * NUM_STAGES];
    __shared__ alignas(16) uint64_t load_mbar_generic[NUM_STAGES];
    __shared__ alignas(16) uint64_t compute_mbar_generic[NUM_STAGES];

    int WARP_ID = threadIdx.x / 32;
    int LANE_ID = threadIdx.x % 32;
    int QUAD_ID = LANE_ID / 4;
    int Q_LANE_ID = LANE_ID % 4;
    int WID_X = WARP_ID / 2;
    int WID_Y = WARP_ID % 2;

    uint32_t smem_base = __cvta_generic_to_shared(smem_generic);
    uint32_t load_mbar = __cvta_generic_to_shared(load_mbar_generic);
    uint32_t compute_mbar = __cvta_generic_to_shared(compute_mbar_generic);

    int off_m = blockIdx.x * BLOCK_M;
    int off_n = blockIdx.y * BLOCK_N;

    if (WARP_ID == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(compute_mbar + i * 8, 32 * 4);
            mbarrier_init(load_mbar + i * 8, 1);
        }
        fence_mbarrier_init();
        for (int i = 0; i < NUM_STAGES; i++)
            mbarrier_arrive_count(compute_mbar + i * 8, 32 * 4);
    }
    __syncthreads();

    uint32_t sA = smem_base;
    uint32_t sB = sA + sA_mem;
    uint32_t sSFA = sB + sB_mem;
    uint32_t sSFB = sSFA + sSFA_mem;

    // offset gC pointer according to warp tiles
    gC += (off_m + WID_X * WARP_M) * N + (off_n + WID_Y * WARP_N);

    // registers and accumulators for mma instruction
    float rC[WARP_M / MMA_M][WARP_N / MMA_N][4] = {};
    uint32_t rA[WARP_M / MMA_M][4];
    uint32_t rB[WARP_N / MMA_N][2];
    uint32_t rSFA[(WARP_M / MMA_M) / 2];
    uint32_t rSFB[(WARP_N / MMA_N) / 4];

    auto load_smem = [&](int iter_k, int stage_id) {
        const int off_k_matrix = iter_k * BLOCK_K;
        const int off_k_sf = iter_k * (BLOCK_M * BLOCK_K / 16);
        const int stride_sf = (K / 16);
        const int stage_offset = stage_id * STAGE_SIZE;
        const int mbar_addr = load_mbar + stage_id * 8;
        tma_load_2d(sA + stage_offset, &tmap_A, mbar_addr, off_k_matrix, off_m);
        tma_load_2d(sB + stage_offset, &tmap_B, mbar_addr, off_k_matrix, off_n);
        tma_load_flat(sSFA + stage_offset, gSFA + off_m * stride_sf + off_k_sf,
                      sSFA_mem, mbar_addr, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        tma_load_flat(sSFB + stage_offset, gSFB + off_n * stride_sf + off_k_sf,
                      sSFB_mem, mbar_addr, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };

    auto load_regs = [&](int mma_k, int stage_id) {
        uint32_t *sfa_ptr =
            (uint32_t *)__cvta_shared_to_generic(sSFA + stage_id * STAGE_SIZE);
        uint32_t *sfb_ptr =
            (uint32_t *)__cvta_shared_to_generic(sSFB + stage_id * STAGE_SIZE);
        uint32_t sA_curr = sA + stage_id * STAGE_SIZE;
        uint32_t sB_curr = sB + stage_id * STAGE_SIZE;
        #pragma unroll
        for (int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m += 2) {
            int row = QUAD_ID + Q_LANE_ID * 8;
            int col = WID_X * 2 + (mma_m / 2);
            rSFA[mma_m / 2] = sfa_ptr[128 * mma_k + col + row * 4];
        }
        #pragma unroll
        for (int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n += 4) {
            int row = QUAD_ID + Q_LANE_ID * 8;
            int col = WID_Y * 2 + (mma_n / 4);
            rSFB[mma_n / 4] = sfb_ptr[128 * mma_k + col + row * 4];
        }
        #pragma unroll
        for (int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++) {
            int col = mma_k * (MMA_K / 2) + (LANE_ID / 16) * 16;
            int row = (mma_m * MMA_M + WID_X * WARP_M + (LANE_ID % 16));
            ldmatrix_m8n8_x4_b16(rA[mma_m], sA_curr + row * (BLOCK_K / 2) + col);
        }
        #pragma unroll
        for (int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++) {
            int col = mma_k * (MMA_K / 2) + (LANE_ID / 8) * 16;
            int row = (mma_n * MMA_N + WID_Y * WARP_N + (LANE_ID % 8));
            ldmatrix_m8n8_x2_b16(rB[mma_n], sB_curr + row * (BLOCK_K / 2) + col);
        }
    };

    auto compute = [&]() {
        #pragma unroll
        for (int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++) {
            #pragma unroll
            for (int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++) {
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
    };

    auto store = [&]() {
        #pragma unroll
        for (int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++) {
            #pragma unroll
            for (int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++) {
                #pragma unroll
                for (int i = 0; i < 2; i++) {
                    int row = mma_m * MMA_M + (LANE_ID / 4) + i * 8;
                    int col = mma_n * MMA_N + (LANE_ID % 4) * 2;
                    streaming_store_f32x2(
                        &gC[row * N + col],
                        reinterpret_cast<float2 *>(rC[mma_m][mma_n])[i]);
                }
            }
        }
    };

    int stage_id = 0;
    uint32_t load_phase = 0;
    uint32_t compute_phase = 0;

    if (WARP_ID < 4) {
        for (int iter_k = 0; iter_k < K / BLOCK_K; iter_k++) {
            mbarrier_wait(load_mbar + stage_id * 8, phase_idx(load_phase, stage_id));
            phase_flip_idx(load_phase, stage_id);
            for (int mma_k = 0; mma_k < BLOCK_K / MMA_K; mma_k++) {
                load_regs(mma_k, stage_id);
                compute();
            }
            mbarrier_arrive(compute_mbar + stage_id * 8);
            stage_id = (stage_id + 1) % NUM_STAGES;
        }
        store();
    } else if (WARP_ID == 4) {
        for (int iter_k = 0; iter_k < K / BLOCK_K; iter_k++) {
            mbarrier_wait(compute_mbar + stage_id * 8, phase_idx(compute_phase, stage_id));
            phase_flip_idx(compute_phase, stage_id);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(load_mbar + stage_id * 8, STAGE_SIZE);
                load_smem(iter_k, stage_id);
            }
            stage_id = (stage_id + 1) % NUM_STAGES;
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
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, global_addr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, global_addr,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

torch::Tensor nvfp4_gemm_v2(
    const torch::Tensor &A,
    const torch::Tensor &B,
    const torch::Tensor &SF_A,
    const torch::Tensor &SF_B
) {
    const int M = A.size(0);
    const int N = B.size(0);
    const int K = A.size(1) * 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor C = torch::empty({M, N}, options);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_STAGES = 2;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M, (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(5 * 32, 1, 1);
    CUtensorMap tma_A = make_tma_descriptor_fp4_A(A.data_ptr(), M, K, BLOCK_M, BLOCK_K);
    CUtensorMap tma_B = make_tma_descriptor_fp4_B(B.data_ptr(), N, K, BLOCK_N, BLOCK_K);
    kernel_v2<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES><<<grid, block>>>(
        SF_A.view(torch::kUInt8).data_ptr<uint8_t>(),
        SF_B.view(torch::kUInt8).data_ptr<uint8_t>(),
        tma_A, tma_B,
        C.data_ptr<float>(),
        M, N, K
    );
    return C;
}