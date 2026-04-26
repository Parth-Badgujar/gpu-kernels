#pragma once
#include <cstdint>

__device__ __forceinline__
void cp_async_ca(uint32_t smem_ptr,
                                            const void *gmem_ptr, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(smem_ptr),
                 "l"(gmem_ptr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__
void cp_async_cg(uint32_t smem_ptr,
                                            const void *gmem_ptr, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" ::"r"(smem_ptr),
                 "l"(gmem_ptr), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__
void cp_async_cg_L2_256B(uint32_t smem_ptr, const void *gmem_ptr, int bytes) {
    asm volatile(
        "cp.async.cg.shared.global.L2::256B [%0], [%1], %2;" ::"r"(smem_ptr),
        "l"(gmem_ptr), "r"(bytes)
        : "memory");
}

__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

__device__ __forceinline__
void cp_async_bulk_commit_group() {
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

template <int N>
__device__ __forceinline__
void cp_async_bulk_wait_group() {
    asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void
cp_async_wait_all() {
    asm volatile("cp.async.wait_all;" ::: "memory");
}

__device__ __forceinline__ void
bar_sync(int bar_id, int num_threads){
    asm volatile("bar.sync %0, %1;\n" ::"r"(bar_id), "r"(num_threads) : "memory");
}