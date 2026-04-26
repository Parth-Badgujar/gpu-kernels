#include <cstdint>
#include <cuda_runtime_api.h>
#include <unistd.h>
#include <iostream>
#include "kernel_headers.cuh"
#include <torch/torch.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "../../include/others.cuh"

auto create_tensors(int M, int N, int K, int sf_vec_size = 16, uint64_t seed = 42) {
    int device_id;
    cudaGetDevice(&device_id);
    auto gen = at::make_generator<at::CUDAGeneratorImpl>(static_cast<c10::DeviceIndex>(device_id));
    gen.set_current_seed(seed);
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA);
    auto a = torch::randint(-128, 127, {M, K / 2}, gen, options).view(torch::kFloat4_e2m1fn_x2);
    auto b = torch::randint(-128, 127, {N, K / 2}, gen, options).view(torch::kFloat4_e2m1fn_x2);
    auto sfa = torch::randint(0, 4, {M, K / sf_vec_size}, gen, options).view(torch::kFloat8_e4m3fn);
    auto sfb = torch::randint(0, 4, {N, K / sf_vec_size}, gen, options).view(torch::kFloat8_e4m3fn);
    int rest_k = K / (sf_vec_size * 4);
    int rest_m = M / 128;
    int rest_n = N / 128;
    sfa = sfa.reshape({rest_m, 4, 32, rest_k, 4}).permute({0, 3, 2, 1, 4});
    sfb = sfb.reshape({rest_n, 4, 32, rest_k, 4}).permute({0, 3, 2, 1, 4});
    sfa = sfa.contiguous().flatten();
    sfb = sfb.contiguous().flatten();
    return std::make_tuple(a, b, sfa, sfb);
}

auto reference_kernel(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& sfa, const torch::Tensor& sfb){
    auto res = torch::_scaled_mm(
        a,
        b.transpose(0, 1),
        sfa,
        sfb,
        std::nullopt,
        std::nullopt,
        torch::kFloat
    );
    return res;
}

float benchmark(auto kernel, int M, int N, int K, int rep = 100, int warmup = 50){
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    int input_size = ((M * K) / 2) + ((N * K) / 2) + ((M * K) / 16) + ((N * K) / 16);
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    int num_input_groups = (input_size >= l2_cache_size * 3) ? 1 : int(l2_cache_size * 3 / input_size) + 1;
    std::vector<torch::Tensor> vA, vB, vSFA, vSFB;
    for(int i = 0; i < num_input_groups; i++){
        auto [a, b, sfa, sfb] = create_tensors(M, N, K);
        vA.push_back(std::move(a));
        vB.push_back(std::move(b));
        vSFA.push_back(std::move(sfa));
        vSFB.push_back(std::move(sfb));
    }

    cudaDeviceSynchronize();
    for(int i = 0; i < warmup; i++){
        int input_group_index = i % num_input_groups;
        kernel(vA[input_group_index], vB[input_group_index], vSFA[input_group_index], vSFB[input_group_index]);
    }

    cudaEventRecord(start_event);
    for(int i = 0; i < rep; i++){
        int input_group_index = i % num_input_groups;
        kernel(vA[input_group_index], vB[input_group_index], vSFA[input_group_index], vSFB[input_group_index]);
    }
    cudaEventRecord(end_event);
    cudaEventSynchronize(end_event);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start_event, end_event);
    return milliseconds / (float)rep;
}

bool correction(auto kernel, int M, int N, int K){
    auto [a, b, sfa, sfb] = create_tensors(M, N, K);
    auto res_ref = reference_kernel(a, b, sfa, sfb);
    auto res_kernel = kernel(a, b, sfa, sfb);
    cudaDeviceSynchronize();
    return torch::allclose(res_ref, res_kernel, 1e-4, 1e-4);
}

int main(){
    auto kernels = std::map<std::string, decltype(&nvfp4_gemm_v1)>{
        {"kernel_v1", &nvfp4_gemm_v1}, 
        {"kernel_v2", &nvfp4_gemm_v2}, 
        {"kernel_v3", &nvfp4_gemm_v3}, 
        {"kernel_v4", &nvfp4_gemm_v4}, 
        {"kernel_v5", &nvfp4_gemm_v5}, 
        {"kernel_v6", &nvfp4_gemm_v6}, 
        {"kernel_v6.5", &nvfp4_gemm_v8},
        {"kernel_v8", &nvfp4_gemm_v9}, 
        {"cuBLAS",    &reference_kernel}
    };
    
    auto run_correction = [&](int M, int N, int K){
        std::cout << "--------- Shape (" << M << ", " << N << ", " << K << ") ---------\n";
        for(auto kernel : kernels){
            bool value = correction(kernel.second, M, N, K);
            std::cout << kernel.first << " : " << value << "\n"; 
        }
    };

    auto run_benchmark = [&](int M, int N, int K){
        std::cout << "--------- Shape (" << M << ", " << N << ", " << K << ") ---------\n";
        for(auto kernel : kernels){
            float time = benchmark(kernel.second, M, N, K);
            sleep(2.0);
            std::cout << kernel.first << " : " << time << " ms\n"; 
        }
    };

    //Some issue with 6.5 version (ignore that)
    run_correction(128, 128, 256);
    run_correction(256, 256, 256);
    run_correction(512, 512, 512);
    run_correction(1024, 1024, 1024);
    run_correction(2048, 2048, 2048);
    run_correction(4096, 4096, 4096);
    run_correction(8192, 8192, 8192);
    run_correction(16384, 16384, 16384);
    run_benchmark(256, 256, 256);
    run_benchmark(512, 512, 512);
    run_benchmark(1024, 1024, 1024);
    run_benchmark(2048, 2048, 2048);
    run_benchmark(4096, 4096, 4096);
    run_benchmark(8192, 8192, 8192);
    run_benchmark(16384, 16384, 16384);
}