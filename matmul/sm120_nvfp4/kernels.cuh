#pragma once
#include <torch/torch.h>

torch::Tensor nvfp4_gemm_v1(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
torch::Tensor nvfp4_gemm_v2(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
torch::Tensor nvfp4_gemm_v3(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
torch::Tensor nvfp4_gemm_v4(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
torch::Tensor nvfp4_gemm_v5(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
// torch::Tensor nvfp4_gemm_v6(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);
template<int M, int N, int K>
torch::Tensor nvfp4_gemm_v6(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb);