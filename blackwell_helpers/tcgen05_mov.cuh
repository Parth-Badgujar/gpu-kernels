#pragma once
#include <cstdint>

__device__ __forceinline__ void tcgen_cp_32x128_warpx4(uint32_t taddr, uint64_t sdesc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr),
        "l"(sdesc)
        : "memory");
}

__device__ __forceinline__ void tcgen_cp_128x128(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" ::"r"(taddr),
                 "l"(sdesc)
                 : "memory");
}

__device__ __forceinline__ void tcgen_cp_128x256(uint32_t taddr, uint64_t sdesc) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" ::"r"(taddr),
                 "l"(sdesc)
                 : "memory");
}

__device__ __forceinline__ void tcgen_ld_32x32_x128(float tmp[], uint32_t addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, "
        "%10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, "
        "%27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, "
        "%61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, "
        "%78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, "
        "%95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, "
        "%110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, "
        "%124, %125, %126, %127}, [%128];"
        : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]),
          "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7]), "=f"(tmp[8]), "=f"(tmp[9]),
          "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]),
          "=f"(tmp[14]), "=f"(tmp[15]), "=f"(tmp[16]), "=f"(tmp[17]),
          "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]),
          "=f"(tmp[22]), "=f"(tmp[23]), "=f"(tmp[24]), "=f"(tmp[25]),
          "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]),
          "=f"(tmp[30]), "=f"(tmp[31]), "=f"(tmp[32]), "=f"(tmp[33]),
          "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]),
          "=f"(tmp[38]), "=f"(tmp[39]), "=f"(tmp[40]), "=f"(tmp[41]),
          "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]),
          "=f"(tmp[46]), "=f"(tmp[47]), "=f"(tmp[48]), "=f"(tmp[49]),
          "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]),
          "=f"(tmp[54]), "=f"(tmp[55]), "=f"(tmp[56]), "=f"(tmp[57]),
          "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]),
          "=f"(tmp[62]), "=f"(tmp[63]), "=f"(tmp[64]), "=f"(tmp[65]),
          "=f"(tmp[66]), "=f"(tmp[67]), "=f"(tmp[68]), "=f"(tmp[69]),
          "=f"(tmp[70]), "=f"(tmp[71]), "=f"(tmp[72]), "=f"(tmp[73]),
          "=f"(tmp[74]), "=f"(tmp[75]), "=f"(tmp[76]), "=f"(tmp[77]),
          "=f"(tmp[78]), "=f"(tmp[79]), "=f"(tmp[80]), "=f"(tmp[81]),
          "=f"(tmp[82]), "=f"(tmp[83]), "=f"(tmp[84]), "=f"(tmp[85]),
          "=f"(tmp[86]), "=f"(tmp[87]), "=f"(tmp[88]), "=f"(tmp[89]),
          "=f"(tmp[90]), "=f"(tmp[91]), "=f"(tmp[92]), "=f"(tmp[93]),
          "=f"(tmp[94]), "=f"(tmp[95]), "=f"(tmp[96]), "=f"(tmp[97]),
          "=f"(tmp[98]), "=f"(tmp[99]), "=f"(tmp[100]), "=f"(tmp[101]),
          "=f"(tmp[102]), "=f"(tmp[103]), "=f"(tmp[104]), "=f"(tmp[105]),
          "=f"(tmp[106]), "=f"(tmp[107]), "=f"(tmp[108]), "=f"(tmp[109]),
          "=f"(tmp[110]), "=f"(tmp[111]), "=f"(tmp[112]), "=f"(tmp[113]),
          "=f"(tmp[114]), "=f"(tmp[115]), "=f"(tmp[116]), "=f"(tmp[117]),
          "=f"(tmp[118]), "=f"(tmp[119]), "=f"(tmp[120]), "=f"(tmp[121]),
          "=f"(tmp[122]), "=f"(tmp[123]), "=f"(tmp[124]), "=f"(tmp[125]),
          "=f"(tmp[126]), "=f"(tmp[127])
        : "r"(addr));
}

__device__ __forceinline__ void tcgen_ld_32x32_x8(float tmp[], uint32_t addr) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, "
                 "%5, %6, %7}, [%8];"
                 : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                   "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                 : "r"(addr));
}

__device__ __forceinline__ void tcgen_ld_wait_sync() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ void tcgen_alloc(uint32_t *tmem_addr_ptr, uint32_t ncols) {
    uint32_t addr = __cvta_generic_to_shared(tmem_addr_ptr);
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(addr), "r"(ncols)
        : "memory");
}

__device__ __forceinline__ void tcgen_delloc(uint32_t taddr, uint32_t ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(taddr), "r"(ncols)
                 : "memory");
}