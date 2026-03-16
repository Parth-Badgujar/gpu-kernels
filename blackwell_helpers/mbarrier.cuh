#pragma once
#include <cstdint>

__device__ inline void mbarrier_wait(uint32_t mbar_addr, uint32_t phase) {
    uint32_t ticks = 0x989680; // this is optional
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, "
                 "[%0], %1, %2;\n\t"
                 "@P1 bra.uni DONE;\n\t"
                 "bra.uni LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}" ::"r"(mbar_addr),
                 "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t mbar_ptr,
                                                          uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
            "r"(mbar_ptr),
        "r"(tx_bytes)
        : "memory");
}

__device__ __forceinline__ void
mbarrier_arrive_expect_tx_cluster(uint32_t mbar_ptr, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster."
                 "b64 _, [%0], %1;" ::"r"(mbar_ptr),
                 "r"(tx_bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx(uint32_t mbar_ptr,
                                                   uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" ::"r"(
            mbar_ptr),
        "r"(tx_bytes)
        : "memory");
}

__device__ __forceinline__ void mbarrier_expect_tx_cluster(uint32_t mbar_ptr,
                                                           uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cluster.shared::cluster.b64 [%0], %1;" ::
            "r"(mbar_ptr),
        "r"(tx_bytes)
        : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint32_t mbar_ptr,
                                              uint32_t thread_count = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_ptr),
                 "r"(thread_count)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint32_t mbar_ptr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(mbar_ptr)
        : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_cluster(uint32_t mbar_ptr) {
    asm volatile(
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(
            mbar_ptr)
        : "memory");
}

// Signal mbarrier that all prior cp.async ops in this thread are done
__device__ __forceinline__ void cp_async_mbarrier_arrive(uint32_t mbar_ptr) {
    asm volatile(
        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(mbar_ptr)
        : "memory");
}

// Same but does not decrement the arrival count of the mbarrier
__device__ __forceinline__ void
cp_async_mbarrier_arrive_noinc(uint32_t mbar_ptr) {
    asm volatile(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(mbar_ptr)
        : "memory");
}

__device__ __forceinline__ uint32_t
mbarrier_get_pending_tx_count(uint64_t state) {
    uint32_t count;
    asm volatile("mbarrier.pending_count.b64 %0, %1;"
                 : "=r"(count)
                 : "l"(state)
                 : "memory");
    return count;
}

__device__ __forceinline__ bool mbarrier_test_wait(uint32_t mbar_ptr,
                                                   uint32_t phase) {
    uint32_t is_ready = 0;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.u32 %0, 1, 0, p;\n\t"
                 "}"
                 : "=r"(is_ready)
                 : "r"(mbar_ptr), "r"(phase)
                 : "memory");
    return is_ready != 0;
}

__device__ __forceinline__ uint64_t mbarrier_arrive_state(uint32_t mbar_ptr){
    uint64_t state;
    uint64_t* ptr = (uint64_t*)__cvta_shared_to_generic(mbar_ptr);
    asm volatile("mbarrier.arrive.noComplete.b64 %0, [%1], 1;" : "=l"(state) : "l"(ptr) : "memory");
    return state;
}