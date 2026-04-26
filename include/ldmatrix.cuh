#pragma once
#include <cstdint>

__device__ __forceinline__
void ldmatrix_m16n16_x2_b8(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.trans.x2.b8 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr)
                 : "memory");
}

__device__ __forceinline__
void ldmatrix_m8n8_x4_trans_b16(uint32_t regs[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
        : "r"(addr));
}

__device__ __forceinline__
void ldmatrix_m8n8_x4_b16(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                 : "r"(addr));
}

__device__ __forceinline__
void ldmatrix_m8n8_x2_b16(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}

__device__ __forceinline__
void ldmatrix_m8n8_x2_trans_b16(uint32_t regs[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
                 : "=r"(regs[0]), "=r"(regs[1])
                 : "r"(addr));
}