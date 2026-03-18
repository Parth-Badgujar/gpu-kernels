#include "block_scaled_nvfp4.cuh"
#include "profiler.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <stdbool.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using fp16 = __half;

template <typename T>
void read_data(const char *filename, thrust::host_vector<T> &data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    file.close();
}

template <typename T>
void write_data(const char *filename, thrust::host_vector<T> &data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file.write(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    file.close();
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./out <M> <N> <K> benchmark|verify" << std::endl;
        return 1;
    }
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    std::string mode = argv[4];
    thrust::host_vector<fp16> c(M * N);
    const thrust::host_vector<uint8_t> sfa(M * K / 16);
    const thrust::host_vector<uint8_t> sfb(N * K / 16);
    const thrust::host_vector<uint8_t> a(M * K / 2);
    const thrust::host_vector<uint8_t> b(N * K / 2);
    read_data("a_scales.bin", const_cast<thrust::host_vector<uint8_t> &>(sfa));
    read_data("b_scales.bin", const_cast<thrust::host_vector<uint8_t> &>(sfb));
    read_data("a_mat.bin", const_cast<thrust::host_vector<uint8_t> &>(a));
    read_data("b_mat.bin", const_cast<thrust::host_vector<uint8_t> &>(b));

    thrust::device_vector<fp16> d_c(M * N);
    const thrust::device_vector<uint8_t> d_sfa = sfa;
    const thrust::device_vector<uint8_t> d_sfb = sfb;
    const thrust::device_vector<uint8_t> d_a = a;
    const thrust::device_vector<uint8_t> d_b = b;

    auto run_bench = [&](auto func) {
        int warmup = 10;
        int rep = 100;
        CUDAtimer timer;
        for (int i = 0; i < warmup; i++) {
            func(d_c.data().get(), d_sfa.data().get(), d_sfb.data().get(),
                 d_a.data().get(), d_b.data().get(), M, N, K);
        }
        timer.start_timer();
        for (int i = 0; i < rep; i++) {
            func(d_c.data().get(), d_sfa.data().get(), d_sfb.data().get(),
                 d_a.data().get(), d_b.data().get(), M, N, K);
        }
        timer.stop_timer();
        return timer.get_time() / (float)rep;
    };

    auto run_verify = [&](auto func, const char *filename) {
        func(d_c.data().get(), d_sfa.data().get(), d_sfb.data().get(),
             d_a.data().get(), d_b.data().get(), M, N, K);
        c = d_c;
        write_data(filename, c);
    };

    if (mode == "verify") {
        run_verify(matmul_nvfp4_v5, "c_out_v5.bin");
        // run_verify(matmul_nvfp4_v4, "c_out_v4.bin");
        // run_verify(matmul_nvfp4_v3, "c_out_v3.bin");
        // run_verify(matmul_nvfp4_v2, "c_out_v2.bin");
    } else if (mode == "benchmark") {
        float t1 = run_bench(matmul_nvfp4_v5);
        float t2 = run_bench(matmul_nvfp4_v4);
        float t3 = run_bench(matmul_nvfp4_v3);
        float t4 = run_bench(matmul_nvfp4_v2);
        std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << "\n";
    } else {
        std::cerr << "Invalid mode" << std::endl;
        return 1;
    }

    return 0;
}