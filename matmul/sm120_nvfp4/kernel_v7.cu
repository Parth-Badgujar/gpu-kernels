#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/torch.h>
#include "../../include/mma_sync.cuh"
#include "../../include/tma.cuh"
#include "../../include/mbarrier.cuh"
#include "../../include/fence.cuh"
#include "../../include/others.cuh"
#include "../../include/ldmatrix.cuh"
#include "../../include/async_group.cuh"

#define MMA_K 64
#define MMA_M 16
#define MMA_N 8

__device__ __forceinline__
int2 get_grid_coords(int bid, int M_BLOCKS, int N_BLOCKS) {
    // constexpr int TILE_SIZE = 8;
    int tiles_per_row = N_BLOCKS >> 3;
    if (tiles_per_row == 0) return make_int2(bid % M_BLOCKS, bid / M_BLOCKS);

    int super_tile_id = bid >> 6;
    int local_id = bid & 63;

    int super_tile_row = super_tile_id / tiles_per_row;
    int super_tile_col = super_tile_id % tiles_per_row;

    int local_row = local_id >> 3;
    int local_col = local_id & 7;

    int m_block = (super_tile_row << 3) + local_row;
    int n_block = (super_tile_col << 3) + local_col;

    if (m_block >= M_BLOCKS || n_block >= N_BLOCKS) {
        m_block = bid % M_BLOCKS;
        n_block = bid / M_BLOCKS;
    }

    return make_int2(m_block, n_block);
}

__device__ __forceinline__
int phase_idx(uint32_t &phase, int idx) {
    return (phase >> idx) & 1;
}

__device__ __forceinline__
void phase_flip_idx(uint32_t &phase, int idx) {
    phase ^= (1 << idx);
}

template<int BLOCK_M = 128, int BLOCK_N = 128, int BLOCK_K = 256, int NUM_STAGES = 2>
__global__ void kernel_v8(
    const uint8_t* __restrict__ gSFA,
    const uint8_t* __restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    const __grid_constant__ CUtensorMap tmap_C,
    int M, int N, int K
){
    constexpr int sA_mem   = BLOCK_M * BLOCK_K / 2;
    constexpr int sB_mem   = BLOCK_N * BLOCK_K / 2;
    constexpr int sSFA_mem = BLOCK_M * BLOCK_K / 16;
    constexpr int sSFB_mem = BLOCK_N * BLOCK_K / 16;
    constexpr int STAGE_SIZE = sA_mem + sB_mem + sSFA_mem + sSFB_mem;

    constexpr int WARP_M = BLOCK_M / 2;
    constexpr int WARP_N = (BLOCK_N / 2) / 2;

    __shared__ alignas(1024) uint8_t smem[STAGE_SIZE * NUM_STAGES];
    __shared__ float sC[2][32 * 72];
    __shared__ uint64_t load_mbar_ptr[NUM_STAGES];;
    __shared__ uint64_t compute_mbar_ptr[NUM_STAGES];


    uint32_t smem_base     = __cvta_generic_to_shared(smem);
    uint32_t sA            = smem_base;
    uint32_t sB            = sA   + sA_mem;
    uint32_t sSFA          = sB   + sB_mem;
    uint32_t sSFB          = sSFA + sSFA_mem;

    uint32_t load_mbar     = __cvta_generic_to_shared(load_mbar_ptr);
    uint32_t compute_mbar   = __cvta_generic_to_shared(compute_mbar_ptr);

    int WARP_ID        = threadIdx.x / 32;
    int WARP_GROUP_ID  = WARP_ID / 4;
    int LANE_ID        = threadIdx.x % 32;
    int QUAD_ID        = LANE_ID / 4;
    int Q_LANE_ID      = LANE_ID % 4;
    int LOCAL_WARP     = WARP_ID % 4;


    uint32_t rA[2][WARP_M / MMA_M][4];
    uint32_t rB[2][WARP_N / MMA_N][2];
    uint32_t rSFA[2][2];
    uint32_t rSFB[2];
    float    rC[WARP_M / MMA_M][WARP_N / MMA_N][4];

    uint32_t load_phase = 0;
    int stage_id = 0;

    if (WARP_ID == 8 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(load_mbar + i * 8, 1);
            mbarrier_init(compute_mbar + i * 8, 32 * 8);
        }
        fence_mbarrier_init();
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_arrive_count(compute_mbar + i * 8, 32 * 8);
        }
    }
    __syncthreads();

    auto load_smem = [&](int iter_k, int stage_id, int off_m, int off_n,
                         uint32_t load_mbar_base) {
        uint32_t lm = load_mbar_base + (stage_id << 3);
        tma_load_2d(sA + stage_id * STAGE_SIZE, &tmap_A, lm,
                    iter_k * BLOCK_K, off_m);
        tma_load_2d(sB + stage_id * STAGE_SIZE, &tmap_B, lm,
                    iter_k * BLOCK_K, off_n);
        tma_load_flat(sSFA + stage_id * STAGE_SIZE,
                      gSFA + off_m * (K / 16) + iter_k * (BLOCK_M * BLOCK_K / 16),
                      sSFA_mem, lm, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
        tma_load_flat(sSFB + stage_id * STAGE_SIZE,
                      gSFB + off_n * (K / 16) + iter_k * (BLOCK_N * BLOCK_K / 16),
                      sSFB_mem, lm, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
    };

    auto load_regs = [&](int mma_k, int stage_id, int reg_phase) {
        uint32_t* sfa_ptr = (uint32_t*)__cvta_shared_to_generic(sSFA + stage_id * STAGE_SIZE);
        uint32_t* sfb_ptr = (uint32_t*)__cvta_shared_to_generic(sSFB + stage_id * STAGE_SIZE);
        uint32_t sA_curr = sA + stage_id * STAGE_SIZE;
        uint32_t sB_curr = sB + stage_id * STAGE_SIZE;
        uint2 data = reinterpret_cast<uint2*>(sfa_ptr + mma_k * 128 + WARP_GROUP_ID * 64)[(QUAD_ID + (Q_LANE_ID & 1) * 8) * 2 + (Q_LANE_ID >> 1)];
        rSFA[reg_phase][0] = data.x;
        rSFA[reg_phase][1] = data.y;
        rSFB[reg_phase]    = (sfb_ptr + mma_k * 128 + LOCAL_WARP * 32 + LANE_ID)[0];


        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            int col = mma_k * (MMA_K / 2) + (LANE_ID / 16) * 16;
            int row = mma_m * (MMA_M * 2) + WARP_GROUP_ID * MMA_M + (LANE_ID % 16);
            ldmatrix_m8n8_x4_b16(rA[reg_phase][mma_m], swizzle<3, 4, 3>(sA_curr + row * (BLOCK_K / 2) + col));
        }

        #pragma unroll
        for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
            int col = mma_k * (MMA_K / 2) + (LANE_ID / 8) * 16;
            int row = mma_n * (MMA_N * 4) + LOCAL_WARP * MMA_N + (LANE_ID % 8);
            ldmatrix_m8n8_x2_b16(rB[reg_phase][mma_n], swizzle<3, 4, 3>(sB_curr + row * (BLOCK_K / 2) + col));
        }
    };

    auto compute = [&](int reg_phase){
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
                mma_m16n8k64_mxf4nvf4_4x_ue4m3(
                    rC[mma_m][mma_n],
                    rA[reg_phase][mma_m],
                    rB[reg_phase][mma_n],
                    rC[mma_m][mma_n],
                    rSFA[reg_phase][mma_m % 2],
                    rSFB[reg_phase],
                    0,
                    mma_m / 2,
                    0,
                    mma_n % 4
                );
            }
        }
    };

    auto epilogue = [&](int off_m, int off_n){
        constexpr int STRIDE = 64 + 8; //padding to remove bank conflicts
        int stage = 0;
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
                int row = WARP_GROUP_ID * MMA_M + (LANE_ID / 4);
                int col = ((mma_n % 2) * (MMA_N * 4) + LOCAL_WARP * MMA_N) + (LANE_ID % 4) * 2;
                int st_n_idx = mma_n / 2;
                float2 data0 = make_float2(rC[mma_m][mma_n][0], rC[mma_m][mma_n][1]);
                float2 data1 = make_float2(rC[mma_m][mma_n][2], rC[mma_m][mma_n][3]);
                reinterpret_cast<float2*>(&sC[stage][row * STRIDE + col])[0] = data0;
                reinterpret_cast<float2*>(&sC[stage][(row + 8) * STRIDE + col])[0] = data1;
                if (mma_n & 1){
                    fence_proxy_async();
                    bar_sync(1, 256);
                    if(WARP_ID == 0 && LANE_ID == 0){
                        tma_store_3d(__cvta_generic_to_shared(sC) + stage * 9216, &tmap_C, 0, (off_n / 64) + st_n_idx, off_m + mma_m * (MMA_M * 2));
                        cp_async_bulk_commit_group();
                        cp_async_bulk_wait_group<1>();
                    }
                    bar_sync(2, 256);
                    stage ^= 1;
                }
            }
        }
    };

    int N_BLOCKS   = N / BLOCK_N;
    int M_BLOCKS   = M / BLOCK_M;
    int NUM_BLOCKS = M_BLOCKS * N_BLOCKS;
    int bid        = blockIdx.x;

    if (WARP_ID < 4) {
        while (bid < NUM_BLOCKS) {
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            memset(rC, 0, sizeof(rC));
            for (int ik = 0; ik < K / BLOCK_K; ik++) {
                mbarrier_wait(load_mbar + (stage_id << 3), phase_idx(load_phase, stage_id));
                phase_flip_idx(load_phase, stage_id);
                load_regs(0, stage_id, 0);
                load_regs(1, stage_id, 1);
                compute(0);
                load_regs(2, stage_id, 0);
                compute(1);
                load_regs(3, stage_id, 1);
                mbarrier_arrive(compute_mbar + (stage_id << 3));
                compute(0);
                compute(1);
                stage_id = (stage_id + 1) % NUM_STAGES;
            }
            epilogue(off_m, off_n);
            bid += gridDim.x;
        }
    }

    else if (WARP_ID < 8) {
        while (bid < NUM_BLOCKS) {
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            memset(rC, 0, sizeof(rC));
            for (int ik = 0; ik < K / BLOCK_K; ik++) {
                mbarrier_wait(load_mbar + (stage_id << 3), phase_idx(load_phase, stage_id));
                phase_flip_idx(load_phase, stage_id);
                load_regs(0, stage_id, 0);
                load_regs(1, stage_id, 1);
                compute(0);
                load_regs(2, stage_id, 0);
                compute(1);
                load_regs(3, stage_id, 1);
                mbarrier_arrive(compute_mbar + (stage_id << 3));
                compute(0);
                compute(1);
                stage_id = (stage_id + 1) % NUM_STAGES;
            }
            epilogue(off_m, off_n);
            bid += gridDim.x;
        }
    }
    else {
        uint32_t cmp_phase = 0;
        if (WARP_ID % 4 == 0){
            while (bid < NUM_BLOCKS) {
                int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
                int off_m = coords.x * BLOCK_M;
                int off_n = coords.y * BLOCK_N;
                for (int ik = 0; ik < K / BLOCK_K; ik++) {
                    mbarrier_wait(compute_mbar + (stage_id << 3), phase_idx(cmp_phase, stage_id));
                    phase_flip_idx(cmp_phase, stage_id);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(load_mbar + (stage_id << 3), STAGE_SIZE);
                        load_smem(ik, stage_id, off_m, off_n, load_mbar);
                    }
                    stage_id = (stage_id + 1) % NUM_STAGES;
                }
                bid += gridDim.x;
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

static CUtensorMap make_tma_descriptor_C(void *global_addr, uint64_t M,
                                         uint64_t N, uint32_t BLOCK_M,
                                         uint32_t BLOCK_N) {
    CUtensorMap tma_desc;
    uint64_t globalDim[3] = {64, N / 64, M};
    uint64_t globalStrides[2] = {64 * 4, N * 4};
    uint32_t boxDim[3] = {64 + 8, 1, 32};
    uint32_t elementStrides[3] = {1, 1, 1};
    cuTensorMapEncodeTiled(
        &tma_desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, global_addr, globalDim,
        globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return tma_desc;
}

torch::Tensor nvfp4_gemm_v8(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& SF_A,
    const torch::Tensor& SF_B
){
    int M = A.size(0);
    int N = B.size(0);
    int K = A.size(1) * 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor c = torch::empty({M, N}, options);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 2;
    dim3 grid(70, 1, 1);
    dim3 block(12 * 32, 1, 1);
    CUtensorMap tma_A = make_tma_descriptor_fp4_A(A.view(torch::kUInt8).data_ptr<uint8_t>(), M, K, BLOCK_M, BLOCK_K);
    CUtensorMap tma_B = make_tma_descriptor_fp4_B(B.view(torch::kUInt8).data_ptr<uint8_t>(), N, K, BLOCK_N, BLOCK_K);
    CUtensorMap tma_C = make_tma_descriptor_C(c.data_ptr<float>(), M, N, BLOCK_M, BLOCK_N);
    kernel_v8<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES><<<grid, block>>>(
        SF_A.view(torch::kUInt8).data_ptr<uint8_t>(),
        SF_B.view(torch::kUInt8).data_ptr<uint8_t>(),
        tma_A, tma_B, tma_C,
        M, N, K
    );
    return c;
}