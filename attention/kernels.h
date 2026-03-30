
#include <cuda_fp16.h>
void attention_simple(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, const size_t B, const 
size_t H, const size_t N, const size_t D);

void attention_v2_tc(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V, __nv_bfloat16* O, const size_t B, const 
size_t H, const size_t N, const size_t D);

