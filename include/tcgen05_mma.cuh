#pragma once
#include <cstdint>

__device__ __forceinline__ uint32_t
make_idesc_mxf4nvf4(uint32_t MMA_M, uint32_t MMA_N, uint32_t transpose_a = 0,
                    uint32_t transpose_b = 0, uint32_t sfa_data_id = 0,
                    uint32_t sfb_data_id = 0) {
    return (0U << 0)                        // bits 0-1:   Reserved = 0
           | (0U << 2)                      // bit 2:      Sparsity = Dense (0)
           | (sfb_data_id << 4)             // bits 4-5:   SFB Data ID = 0
           | (1U << 7)                      // bits 7-9:   atype E2M1 = 1
           | (1U << 10)                     // bits 10-11: btype E2M1 = 1
           | (0U << 13)                     // bit 13:     Negate A = 0
           | (0U << 14)                     // bit 14:     Negate B = 0
           | (transpose_a << 15)            // bit 15:     Transpose A = 0
           | (transpose_b << 16)            // bit 16:     Transpose B = 0
           | ((uint32_t)(MMA_N >> 3) << 17) // bits 17-22: N >> 3
           | (0U << 23) // bit 23:     Scale Type UE4M3 = 0 (mxf4nvf4)
           | ((uint32_t)(MMA_M >> 7) << 27) // bits 27-28: M >> 7
           | (sfa_data_id << 29)            // bits 29-30: SFA Data ID = 0
           | (0U << 31); // bit 31:     K dim = 0 (Dense K=64)
}

__device__ __forceinline__ uint64_t make_smem_descriptor(uint32_t smem_addr,
                                                uint32_t sbo, uint32_t lbo = 16,
                                                uint8_t swizzle_mode = 0) {
    uint64_t desc = 0;
    desc |= ((smem_addr >> 4) & 0x3FFF);
    desc |= (((uint64_t)(lbo >> 4) & 0x3FFF) << 16);
    desc |= (((uint64_t)(sbo >> 4) & 0x3FFF) << 32);
    desc |= ((uint64_t)1 << 46);
    desc |= ((uint64_t)(swizzle_mode & 0x7) << 61);
    return desc;
}

// NVFP4
__device__ __forceinline__ void
tcgen05_mma_mxf4nvf4_4x(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                        uint32_t idesc, uint32_t scale_A_tmem,
                        uint32_t scale_B_tmem, uint32_t enable_input_d) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.u32 p, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::mxf4nvf4"
                 ".block_scale.scale_vec::4X"
                 " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
                 "}\n"
                 :
                 : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
                   "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
                 : "memory");
}

__device__ __forceinline__ void tcgen_commit_arrive_one(uint32_t mbar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                 "cluster.b64 [%0];" ::"r"(mbar)
                 : "memory");
}