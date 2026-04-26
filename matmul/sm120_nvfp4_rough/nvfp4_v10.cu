#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda.h>
#include <string.h>
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

__device__ __forceinline__ int2 get_grid_coords(int bid, int M_BLOCKS, int N_BLOCKS) {
    constexpr int TILE_SIZE = 8; 
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

template<int BLOCK_M = 128, int BLOCK_N = 128, int BLOCK_K = 256, int NUM_STAGES = 2>
__global__ void __launch_bounds__(384, 1) _nvfp4_gemm_v10(
    const uint8_t* __restrict__ gSFA,
    const uint8_t* __restrict__ gSFB,
    const __grid_constant__ CUtensorMap tmap_A,
    const __grid_constant__ CUtensorMap tmap_B,
    const __grid_constant__ CUtensorMap tmap_C,
    int M, int N, int K
    // intra_kernel_profiler::trace::GlobalBuffer prof
){
    constexpr int sA_mem   = BLOCK_M * BLOCK_K / 2;       
    constexpr int sB_mem   = BLOCK_N * BLOCK_K / 2;              
    constexpr int sSFA_mem = BLOCK_M * BLOCK_K / 16;             
    constexpr int sSFB_mem = BLOCK_N * BLOCK_K / 16;             
    constexpr int STAGE_SIZE = sA_mem + sB_mem + sSFA_mem + sSFB_mem;
    constexpr int GROUP_M = 2;
    constexpr int GROUP_N = 2;
    constexpr int WARP_M = BLOCK_M / GROUP_M;                   
    constexpr int WARP_N = BLOCK_N / GROUP_N;

    __shared__ uint8_t smem[STAGE_SIZE * NUM_STAGES + (8 * NUM_STAGES) * 2 + 8 + 8];
    __shared__ alignas(128) float sC[2][32 * (64 + 8)];
    // IKP_TRACE_CTX_TYPE(4096, 5) ctx;
    // IKP_TRACE_CTX_INIT(ctx);
    
    uint32_t smem_base     = __cvta_generic_to_shared(smem);
    uint32_t sA            = smem_base;
    uint32_t sB            = sA   + sA_mem;
    uint32_t sSFA          = sB   + sB_mem;
    uint32_t sSFB          = sSFA + sSFA_mem;
    uint32_t load_mbar     = sSFB + sSFB_mem;
    uint32_t compute_mbar  = load_mbar + NUM_STAGES * 8;
    uint32_t pp0           = compute_mbar + NUM_STAGES * 8;
    uint32_t pp1           = pp0 + 8;
    int warp_id         = threadIdx.x / 32;
    int lane_id         = threadIdx.x % 32;
    int quad_id         = lane_id / 4;
    int lane_id_in_quad = lane_id % 4;
    int local_warp_id   = (warp_id % 4);
    int wid_x           = local_warp_id / GROUP_N;
    int wid_y           = local_warp_id % GROUP_N;

    uint32_t rA[WARP_M / MMA_M][4];
    uint32_t rB[WARP_N / MMA_N][2];
    uint32_t rSFA[4 / GROUP_M];
    uint32_t rSFB[4 / GROUP_N];
    float    rC[WARP_M / MMA_M][WARP_N / MMA_N][4];
    float    zero[4] = {0.0f};

    int stage_id = 0;
    if (warp_id == 0 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(load_mbar + i * 8, 1);
            mbarrier_init(compute_mbar + i * 8, 32 * 4);      
        }
        mbarrier_init(pp0, 32 * 4);
        mbarrier_init(pp1, 32 * 4);
        fence_mbarrier_init();
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_arrive_count(compute_mbar + i * 8, 32 * 4);
        }
        mbarrier_arrive_count(pp1, 128);
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

    auto load_regs = [&](int mma_k, int stage_id) {
        uint32_t* sfa_ptr = (uint32_t*)__cvta_shared_to_generic(sSFA + stage_id * STAGE_SIZE);
        uint32_t* sfb_ptr = (uint32_t*)__cvta_shared_to_generic(sSFB + stage_id * STAGE_SIZE);
        uint32_t sA_curr = sA + stage_id * STAGE_SIZE;
        uint32_t sB_curr = sB + stage_id * STAGE_SIZE;
        uint2 data = reinterpret_cast<uint2*>(sfa_ptr + mma_k * 128 + wid_x * 64)[(quad_id + (lane_id_in_quad & 1) * 8) * 2 + (lane_id_in_quad >> 1)];
        rSFA[0] = data.x;
        rSFA[1] = data.y;
        int off_sfb = mma_k * 128 + wid_y * 32 + lane_id;
        rSFB[0] = (sfb_ptr + off_sfb)[0];
        rSFB[1] = (sfb_ptr + off_sfb + 64)[0];


        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            int col = mma_k * (MMA_K / 2) + (lane_id / 16) * 16;
            int row = mma_m * (MMA_M * GROUP_M) + wid_x * MMA_M + (lane_id % 16);
            ldmatrix_m8n8_x4_b16(rA[mma_m], swizzle<3, 4, 3>(sA_curr + row * (BLOCK_K / 2) + col));
        }

        #pragma unroll
        for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
            int col = mma_k * (MMA_K / 2) + (lane_id / 8) * 16;
            int row = mma_n * (MMA_N * GROUP_N) + wid_y * MMA_N + (lane_id % 8);
            ldmatrix_m8n8_x2_b16(rB[mma_n], swizzle<3, 4, 3>(sB_curr + row * (BLOCK_K / 2) + col));
        }
    };

    auto compute = [&](bool acc){
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){
                mma_m16n8k64_mxf4nvf4_4x_ue4m3(
                    rC[mma_m][mma_n],
                    rA[mma_m],
                    rB[mma_n],
                    acc ? rC[mma_m][mma_n] : zero,
                    rSFA[mma_m % 2],
                    rSFB[mma_n % 2],
                    0,
                    mma_m / 2,
                    0,
                    mma_n / 2
                );
            }
        }
    };

    auto epilogue = [&](int off_m, int off_n){
        constexpr int STRIDE = 64 + 8;
        int stage = 0;
        #pragma unroll
        for(int mma_m = 0; mma_m < WARP_M / MMA_M; mma_m++){
            #pragma unroll
            for(int mma_n = 0; mma_n < WARP_N / MMA_N; mma_n++){    
                int row = wid_x * MMA_M + (lane_id / 4);
                int col = ((mma_n % 4) * (MMA_N * GROUP_N) + wid_y * MMA_N) + 2 * (lane_id % 4);
                int st_n_idx = mma_n / 4;
                float2 data0 = make_float2(rC[mma_m][mma_n][0], rC[mma_m][mma_n][1]);
                float2 data1 = make_float2(rC[mma_m][mma_n][2], rC[mma_m][mma_n][3]);
                reinterpret_cast<float2*>(&sC[stage][row * STRIDE + col])[0] = data0;
                reinterpret_cast<float2*>(&sC[stage][(row + 8) * STRIDE + col])[0] = data1;
                if (mma_n % 4 == 3){
                    fence_proxy_async();
                    bar_sync(1 + (warp_id / 4) * 2, 128);
                    if(local_warp_id == 0 && lane_id == 0){
                        tma_store_3d(__cvta_generic_to_shared(sC) + stage * 32 * (64 + 8) * 4, &tmap_C, 0, (off_n / 64) + st_n_idx, off_m + mma_m * (MMA_M * GROUP_M));
                        cp_async_bulk_commit_group();
                        cp_async_bulk_wait_group<1>();
                    }
                    bar_sync(2 + (warp_id / 4) * 2, 128);
                    stage ^= 1;
                }
            }
        }
    };

    int N_BLOCKS   = N / BLOCK_N;
    int M_BLOCKS   = M / BLOCK_M;
    int NUM_BLOCKS = M_BLOCKS * N_BLOCKS;

    if (warp_id < 4) {
        int bid = blockIdx.x + gridDim.x * (warp_id / 4);
        setmaxnreg_inc<232>();
        uint32_t load_phase = 0;
        uint32_t pp_phase = 0;
        while (bid < NUM_BLOCKS) {
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            memset(rC, 0, sizeof(rC));
            mbarrier_wait(pp1, pp_phase);
            pp_phase ^= 1;

            for (int ik = 0; ik < K / BLOCK_K; ik++) {
                mbarrier_wait(load_mbar + (stage_id << 3), (load_phase >> stage_id) & 1);
                // IKP_TRACE_REC_B(ctx, prof, 0);
                load_phase ^= (1 << stage_id);
                load_regs(0, stage_id);
                compute(ik != 0);
                load_regs(1, stage_id);
                compute(true);
                load_regs(2, stage_id);
                compute(true);
                load_regs(3, stage_id);
                mbarrier_arrive(compute_mbar + (stage_id << 3));
                compute(true);
                stage_id = (stage_id + 1) % NUM_STAGES;
                // IKP_TRACE_REC_E(ctx, prof, 0);
            }
            mbarrier_arrive(pp0);
            // IKP_TRACE_REC_B(ctx, prof, 2);
            epilogue(off_m, off_n);
            // IKP_TRACE_REC_E(ctx, prof, 2);
            bid += 2 * gridDim.x;
        }
    }
    else if (warp_id < 8){
        int bid = blockIdx.x + (warp_id / 4) * gridDim.x;
        setmaxnreg_inc<232>();
        uint32_t load_phase = 0;
        uint32_t pp_phase = 0;
        while (bid < NUM_BLOCKS) {
            int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
            int off_m = coords.x * BLOCK_M;
            int off_n = coords.y * BLOCK_N;
            memset(rC, 0, sizeof(rC));
            mbarrier_wait(pp0, pp_phase);
            pp_phase ^= 1;
            for (int ik = 0; ik < K / BLOCK_K; ik++) {
                mbarrier_wait(load_mbar + (stage_id << 3), (load_phase >> stage_id) & 1);
                // IKP_TRACE_REC_B(ctx, prof, 0);
                load_phase ^= (1 << stage_id);
                load_regs(0, stage_id);
                compute(ik != 0);
                load_regs(1, stage_id);
                compute(true);
                load_regs(2, stage_id);
                compute(true);
                load_regs(3, stage_id);
                mbarrier_arrive(compute_mbar + (stage_id << 3));
                compute(true);
                stage_id = (stage_id + 1) % NUM_STAGES;
                // IKP_TRACE_REC_E(ctx, prof, 0);
            }
            mbarrier_arrive(pp1);
            // IKP_TRACE_REC_B(ctx, prof, 2);
            epilogue(off_m, off_n);
            // IKP_TRACE_REC_E(ctx, prof, 2);
            bid += 2 * gridDim.x;
        }
    }
    else {
        int bid = blockIdx.x;
        setmaxnreg_dec<32>();
        uint32_t cmp_phase = 0;
        if (local_warp_id == 0){
            while (bid < NUM_BLOCKS) {
                int2 coords = get_grid_coords(bid, M_BLOCKS, N_BLOCKS);
                int off_m = coords.x * BLOCK_M;
                int off_n = coords.y * BLOCK_N;
                for (int ik = 0; ik < K / BLOCK_K; ik++) {
                    mbarrier_wait(compute_mbar + (stage_id << 3), (cmp_phase >> stage_id) & 1);
                    // IKP_TRACE_REC_B(ctx, prof, 1);
                    cmp_phase ^= (1 << stage_id);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(load_mbar + (stage_id << 3), STAGE_SIZE);
                        load_smem(ik, stage_id, off_m, off_n, load_mbar);
                    }
                    stage_id = (stage_id + 1) % NUM_STAGES;
                    // IKP_TRACE_REC_E(ctx, prof, 1);
                }
                bid += gridDim.x;
            }
        }
    }
    // IKP_TRACE_CTX_FLUSH(ctx, prof);
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

torch::Tensor nvfp4_gemm_v10(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb){
    int M = a.size(0);
    int N = b.size(0);
    int K = a.size(1) * 2;
    // intra_kernel_profiler::trace::HostSession sess;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor c = torch::empty({M, N}, options);
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 256;
    constexpr int NUM_STAGES = 2;
    dim3 grid(70, 1, 1);
    dim3 block(12 * 32, 1, 1);
    // sess.set_region_names({"compute", "load", "epilogue"});
    // sess.set_block_filter({0, 30});
    // sess.init(/*cap=*/4096, /*grid_x=*/grid.x, /*threads_per_block=*/block.x);
    // sess.reset();
    CUtensorMap tma_A = make_tma_descriptor_fp4_A(a.view(torch::kUInt8).data_ptr<uint8_t>(), M, K, BLOCK_M, BLOCK_K);
    CUtensorMap tma_B = make_tma_descriptor_fp4_B(b.view(torch::kUInt8).data_ptr<uint8_t>(), N, K, BLOCK_N, BLOCK_K);
    CUtensorMap tma_C = make_tma_descriptor_C(c.data_ptr<float>(), M, N, BLOCK_M, BLOCK_N);
    _nvfp4_gemm_v10<BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES><<<grid, block>>>(
        sfa.view(torch::kUInt8).data_ptr<uint8_t>(),
        sfb.view(torch::kUInt8).data_ptr<uint8_t>(), 
        tma_A, tma_B, tma_C, 
        M, N, K
        // sess.global_buffer()
    );
    // cudaDeviceSynchronize();
    // sess.write_trace("nvfp4_v10.json");
    return c;
}