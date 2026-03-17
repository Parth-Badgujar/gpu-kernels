#pragma once
#include <cstdint>

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ __forceinline__
void store256(void* ptr, uint32_t* src)
{
    asm volatile(
        "st.global.L1::no_allocate.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
        :
        : "l"(ptr),
          "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
          "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7])
        : "memory"
    );
}

__device__ __forceinline__
void load256(void* ptr, uint32_t* src)
{
    asm volatile(
        "ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
        :
        : "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
          "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]),
          "l"(ptr)
        : "memory"
    );
}