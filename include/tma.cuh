#pragma once
#include <cstdint>

__device__ __forceinline__
void tma_load_flat(
    uint32_t dst, const void *src,
    int size, int mbar_addr,
    uint64_t cache_policy
) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::"
                 "bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;" ::"r"(dst),
                 "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__
void tma_load_1d(
    uint32_t dst, const void *tmap_ptr,
    int mbar_addr, int x
) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::"
                 "complete_tx::bytes [%0], [%1, {%2}], [%3];" ::"r"(dst),
                 "l"(tmap_ptr), "r"(x), "r"(mbar_addr)
                 : "memory");
}

__device__ __forceinline__
void tma_load_2d(
    uint32_t dst, const void *tmap_ptr,
    int mbar_addr, int x, int y
) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::"
                 "complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
                 "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr)
                 : "memory");
}

__device__ __forceinline__
void tma_load_3d(
    uint32_t dst, const void *tmap_ptr,
    int mbar_addr, int x, int y, int z
) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
        "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(dst),
        "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_load_4d(
    uint32_t dst, const void *tmap_ptr,
    int mbar_addr, int x, int y, int z, int w
) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::"
        "bytes [%0], [%1, {%2, %3, %4, %5}], [%6];" ::"r"(dst),
        "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_load_5d(
    uint32_t dst, const void *tmap_ptr,
    int mbar_addr, int x, int y, int z, int w, int v
) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::"
        "bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::"r"(dst),
        "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v), "r"(mbar_addr)
        : "memory");
}

__device__ __forceinline__
void tma_store_flat(
    void* dst, uint32_t src, int size
) {
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
        :: "l"(dst), "r"(src), "r"(size)
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_2d(
    uint32_t src, const void* tmap_ptr, int x, int y
) {
    asm volatile (
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%1, {%2, %3}], [%0];"
        :: "r"(src), "l"(tmap_ptr), "r"(x), "r"(y)
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_3d(
    uint32_t src, const void* tmap_ptr, int x, int y, int z
) {
    asm volatile (
        "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%1, {%2, %3, %4}], [%0];"
        :: "r"(src), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z)
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_4d(
    uint32_t src, const void* tmap_ptr, int x, int y, int z, int w
) {
    asm volatile (
        "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%1, {%2, %3, %4, %5}], [%0];"
        :: "r"(src), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w)
        : "memory"
    );
}

__device__ __forceinline__
void tma_store_5d(
    uint32_t src, const void* tmap_ptr, int x, int y, int z, int w, int v
) {
    asm volatile (
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%1, {%2, %3, %4, %5, %6}], [%0];"
        :: "r"(src), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v)
        : "memory"
    );
}