#pragma once
#include <stdint.h>

__device__ __forceinline__
void clc_try_cancel(uint32_t result, uint32_t mbar)
{
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];\n\t"
        :                                      // no outputs – result written async
        : "r"(result),               // %0 : result buffer address
          "r"(mbar)                  // %1 : mbarrier address
        : "memory"                             // async write to shared memory
    );
}

__device__ __forceinline__
bool clc_query_is_canceled(uint4 result)
{
    // The b128 operand maps to a quad of u32 registers {r0,r1,r2,r3}.
    uint32_t pred_u32 = 0;
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  .reg .b128 result128;\n\t"
        "  mov.b128 result128, {%1, %2, %3, %4};\n\t"
        "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128"
        "    p, result128;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(pred_u32)                       // %0 : 1=canceled, 0=not
        : "r"(result.x),                       // %1 : b128 word 0
          "r"(result.y),                       // %2 : b128 word 1
          "r"(result.z),                       // %3 : b128 word 2
          "r"(result.w)                        // %4 : b128 word 3
    );
    return static_cast<bool>(pred_u32);
}

__device__ __forceinline__
int clc_query_ctaid_x(uint4 result)
{
    int bx = 0;
    asm volatile(
        "{\n\t"
        "  .reg .b128 result128;\n\t"
        "  mov.b128 result128, {%1, %2, %3, %4};\n\t"
        "  clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128"
        "    %0, result128;\n\t"
        "}\n\t"
        : "=r"(bx)                             // %0 : ctaid.x (s32)
        : "r"(result.x),                       // %1
          "r"(result.y),                       // %2
          "r"(result.z),                       // %3
          "r"(result.w)                        // %4
    );
    return bx;
}

__device__ __forceinline__
int clc_query_ctaid_y(uint4 result)
{
    int by = 0;
    asm volatile(
        "{\n\t"
        "  .reg .b128 result128;\n\t"
        "  mov.b128 result128, {%1, %2, %3, %4};\n\t"
        "  clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128"
        "    %0, result128;\n\t"
        "}\n\t"
        : "=r"(by)                             // %0 : ctaid.y (s32)
        : "r"(result.x),
          "r"(result.y),
          "r"(result.z),
          "r"(result.w)
    );
    return by;
}


__device__ __forceinline__
int clc_query_ctaid_z(uint4 result)
{
    int bz = 0;
    asm volatile(
        "{\n\t"
        "  .reg .b128 result128;\n\t"
        "  mov.b128 result128, {%1, %2, %3, %4};\n\t"
        "  clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128"
        "    %0, result128;\n\t"
        "}\n\t"
        : "=r"(bz)                             // %0 : ctaid.z (s32)
        : "r"(result.x),
          "r"(result.y),
          "r"(result.z),
          "r"(result.w)
    );
    return bz;
}
