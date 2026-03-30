#pragma once
#include <cstdint>

__device__ __forceinline__
void mma_m16n8k64_mxf4nvf4_4x_ue4m3(
    float    d[4],
    uint32_t a[4],
    uint32_t b[2],
    float    c[4],
    uint32_t sfa,
    uint32_t sfb)
{
    constexpr uint16_t BID_A = 0, TID_A = 0;
    constexpr uint16_t BID_B = 0, TID_B = 0;

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14}, {%15, %16},"
        "{%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
           "r"(sfa), "h"(BID_A), "h"(TID_A),
           "r"(sfb), "h"(BID_B), "h"(TID_B)
    );
}

__device__ inline
void mma_m16n8k16_row_col_f32_bf16_bf16_f32(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}
