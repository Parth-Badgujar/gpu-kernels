#pragma once
#include <cstdint>
#include <cuda_fp16.h>

using fp16 = __half;

void matmul_nvfp4_v1(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K);
void matmul_nvfp4_v2(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K);
void matmul_nvfp4_v3(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K);
void matmul_nvfp4_v4(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K);
void matmul_nvfp4_v5(fp16 *c, const uint8_t *sfa, const uint8_t *sfb,
                     const uint8_t *a, const uint8_t *b, int M, int N, int K);
