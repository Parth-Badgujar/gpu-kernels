#pragma once
#include <cstdint>
// ─────────────────────────────────────────────────────────────────
// cp.async — element-wise global → shared (SM_80+)
// All smem pointers are pre-converted uint32_t addresses
// ─────────────────────────────────────────────────────────────────

// cache in L1+L2
__device__ __forceinline__ void cp_async_ca(uint32_t smem_ptr,
                                            const void *gmem_ptr, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(smem_ptr),
                 "l"(gmem_ptr), "r"(bytes)
                 : "memory");
}

// cache in L2 only
__device__ __forceinline__ void cp_async_cg(uint32_t smem_ptr,
                                            const void *gmem_ptr, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem_ptr),
                 "l"(gmem_ptr), "r"(bytes)
                 : "memory");
}

// bypass L1, 256B L2 cache hint
__device__ __forceinline__ void
cp_async_cg_L2_256B(uint32_t smem_ptr, const void *gmem_ptr, int bytes) {
    asm volatile(
        "cp.async.cg.shared.global.L2::256B [%0], [%1], %2;" ::"r"(smem_ptr),
        "l"(gmem_ptr), "r"(bytes)
        : "memory");
}

// ─────────────────────────────────────────────────────────────────
// async_group completion (legacy, SM_80+)
// ─────────────────────────────────────────────────────────────────

__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// Leave N most-recent groups still in flight, wait for the rest
template <int N> __device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}