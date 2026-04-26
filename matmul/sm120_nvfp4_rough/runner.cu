#include <cstdint>
#include <vector>
#include <optional>
#include <iostream>
#include <unistd.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include "kernels.cuh"
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


auto reference_kernel(torch::Tensor a, torch::Tensor b, torch::Tensor sfa, torch::Tensor sfb){
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

float benchmark(auto kernel, int M, int N, int K, int rep = 1000, int warmup = 500){
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

int main(int argc, char* argv[]){
    if (argc != 4){
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    auto [a, b, sfa, sfb] = create_tensors(M, N, K);
    if (1){
        auto c = reference_kernel(a, b, sfa, sfb);
        // auto c6 = nvfp4_gemm_v6(a, b, sfa, sfb);
        // auto c7 = nvfp4_gemm_v7(a, b, sfa, sfb);
        // auto c8 = nvfp4_gemm_v8(a, b, sfa, sfb);
        // auto c9 = nvfp4_gemm_v9(a, b, sfa, sfb);
        auto c10 = nvfp4_gemm_v9(a, b, sfa, sfb);
        CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << c10.view({-1, 128})[0] << "\n";
        // std::cout << c.view({-1, 128})[0] << "\n";
        // std::cout << torch::allclose(c, c10) << "\n";
        // torch::save(c, "ref.pt");
        // torch::save(c10, "nvfp4_v10.pt");
        // for(int i = 0; i < 6; i++){
        //     nvfp4_gemm_v6(a, b, sfa, sfb);
        // }
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        // for(int i = 0; i < 6; i++){
        //     nvfp4_gemm_v7(a, b, sfa, sfb);
        // }
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
    }
    if (0){
        // float perf_v1 = benchmark(nvfp4_gemm_v1, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        // float perf_v2 = benchmark(nvfp4_gemm_v2, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        // float perf_v3 = benchmark(nvfp4_gemm_v3, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        // float perf_v4 = benchmark(nvfp4_gemm_v4, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        // float perf_v5 = benchmark(nvfp4_gemm_v5, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // sleep(3.0);
        float perf_v6 = benchmark(nvfp4_gemm_v6, M, N, K, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        sleep(3.0);
        float perf_ref = benchmark(reference_kernel, M, N, K, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        sleep(3.0);
        // float perf_v7 = benchmark(nvfp4_gemm_v7, M, N, K, 100);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // // sleep(3.0);
        float perf_v8 = benchmark(nvfp4_gemm_v8, M, N, K, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        sleep(3.0);
        float perf_v9 = benchmark(nvfp4_gemm_v9, M, N, K, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        sleep(3.0);
        float perf_v10 = benchmark(nvfp4_gemm_v10, M, N, K, 100);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "Performance Ref: " << perf_ref << " ms\n";
        // // std::cout << "Performance V1: " << perf_v1 << " ms\n";
        // // std::cout << "Performance V2: " << perf_v2 << " ms\n";
        // // std::cout << "Performance V3: " << perf_v3 << " ms\n";
        // // std::cout << "Performance V4: " << perf_v4 << " ms\n";
        // // std::cout << "Performance V5: " << perf_v5 << " ms\n";
        std::cout << "Performance V6: " << perf_v6 << " ms\n";
        // std::cout << "Performance V7: " << perf_v7 << " ms\n";
        std::cout << "Performance V8: " << perf_v8 << " ms\n";
        std::cout << "Performance V9: " << perf_v9 << " ms\n";
        std::cout << "Performance V10: " << perf_v10 << " ms\n";
    }
}