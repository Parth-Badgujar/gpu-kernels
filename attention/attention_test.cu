#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>
#include <utils.h>
#include <stdlib.h>
#include <cooperative_groups.h>
#include <kernels.h>
#include <omp.h>
#include <cmath>
#include <cuda_bf16.h>
#include <algorithm>
#include <limits>
#include <cfloat>
#include <iomanip>
#include <string>

typedef nv_bfloat16 bf16;
    
void attention_cpu(const float* Q, const float* K, const float* V, float* O,
                   size_t B, size_t H, size_t N, size_t D)
{
    const size_t stride_b = H * N * D;
    const size_t stride_h = N * D;
    const float  neg_inf  = -std::numeric_limits<float>::infinity();
    const float  scale    = 1.0f / sqrtf((float)D);

    float* A = new float[B * H * N * N];

    // phase 1: A[b,h,i,j] = (Q·K^T) * scale
    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b = 0; b < B; ++b)
      for (size_t h = 0; h < H; ++h)
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j < N; ++j) {
            const size_t qbh = b * stride_b + h * stride_h;
            const size_t Kbh = qbh;
            const size_t Abh = b * (H * N * N) + h * (N * N);
            const float* q = Q + qbh + i * D;
            const float* k = K + Kbh + j * D;
            float acc = 0.f;
            for (size_t d = 0; d < D; ++d) acc += q[d] * k[d];
            A[Abh + i * N + j] = acc * scale;
          }

    // phase 2: softmax rows of A
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < B; ++b)
      for (size_t h = 0; h < H; ++h)
        for (size_t i = 0; i < N; ++i) {
          const size_t Abh = b * (H * N * N) + h * (N * N);
          float mx = neg_inf;
          for (size_t j = 0; j < N; ++j) mx = std::max(mx, A[Abh + i * N + j]);
          float den = 0.f;
          for (size_t j = 0; j < N; ++j) {
            float e = std::exp(A[Abh + i * N + j] - mx);
            A[Abh + i * N + j] = e;
            den += e;
          }
          float inv_den = 1.f / den;
          for (size_t j = 0; j < N; ++j) A[Abh + i * N + j] *= inv_den;
        }

    // phase 3: O = A · V
    #pragma omp parallel for collapse(4) schedule(static)
    for (size_t b = 0; b < B; ++b)
      for (size_t h = 0; h < H; ++h)
        for (size_t i = 0; i < N; ++i)
          for (size_t d = 0; d < D; ++d) {
            const size_t Obh = b * stride_b + h * stride_h;
            const size_t Vbh = Obh;
            const size_t Abh = b * (H * N * N) + h * (N * N);
            const float* arow = A + Abh + i * N;
            float acc = 0.f;
            for (size_t k = 0; k < N; ++k)
              acc += arow[k] * V[Vbh + k * D + d];
            O[Obh + i * D + d] = acc;
          }

    delete[] A;
}

int check_output(thrust::host_vector<float>& data1, thrust::host_vector<float>& data2, size_t n){
    thrust::host_vector<float> data1_cpu = data1;
    for(size_t i = 0; i < n; i++){
        float diff = fabs(data1_cpu[i] - data2[i]);
        if (diff > 0.1f){
            std::cout << "CPU : " << data2[i] << " GPU : " << data1_cpu[i] << "\n";
            return 0;
        }
    }
    return 1;
}

typedef void(*kernel_t)(const bf16*, const bf16*, const bf16*, bf16*, const size_t, const size_t, const size_t, const size_t);
// typedef void(*kernel_t)(float*, float*, float*, float*, size_t, size_t, size_t, size_t);

struct config_t { size_t B, H, N, D; };

struct bench_result_t {
    float  time_ms;
    double flops;
    double gflops_per_s;
};

static inline double flops_attention(size_t B, size_t H, size_t N, size_t D){
    return (double)B * (double)H * (4.0 * (double)N * (double)N * (double)D + 5.0 * (double)N * (double)N);
}

bench_result_t benchmark_ex(
    kernel_t kernel,
    bf16* Q, bf16* K, bf16* V, bf16* O,
    size_t B, size_t H, size_t N, size_t D,
    int warmup, int rep,
    double (*flops_fn)(size_t,size_t,size_t,size_t) = flops_attention)
{
    for(int it = 0; it < warmup; it++) kernel(Q, K, V, O, B, H, N, D);
    CUDAtimer *timer = new CUDAtimer();
    timer->start_timer();
    for(int it = 0; it < rep; it++) kernel(Q, K, V, O, B, H, N, D);
    timer->stop_timer();
    float avg_ms = timer->get_time() / (float)rep;
    delete timer;

    double flops = flops_fn(B, H, N, D);
    double gflops_per_s = (flops / (avg_ms / 1000.0)) / 1e9;
    return {avg_ms, flops, gflops_per_s};
}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <cuda_bf16.h>   // __nv_bfloat16, __float2bfloat16_rn, __bfloat162float

using bf16 = __nv_bfloat16;

// assume: kernel takes bf16 pointers now
using kernel_t = void(*)(const bf16*, const bf16*, const bf16*, bf16*,
                         const size_t, const size_t, const size_t, const size_t);

int test_kernel_once(kernel_t kernel, size_t B, size_t H, size_t N, size_t D){
    const size_t T = B * H * N * D;

    thrust::default_random_engine rng(69);
    thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    thrust::host_vector<bf16> q_bf16_h(T), k_bf16_h(T), v_bf16_h(T);
    thrust::generate(q_bf16_h.begin(), q_bf16_h.end(), [&]{return dist(rng);});
    thrust::generate(k_bf16_h.begin(), k_bf16_h.end(), [&]{return dist(rng);});
    thrust::generate(v_bf16_h.begin(), v_bf16_h.end(), [&]{return dist(rng);});
    thrust::device_vector<bf16> q_bf16_d = q_bf16_h;
    thrust::device_vector<bf16> k_bf16_d = k_bf16_h;
    thrust::device_vector<bf16> v_bf16_d = v_bf16_h;
    thrust::device_vector<bf16> o_bf16_d(T); 
    thrust::host_vector<float> q_h(T), k_h(T), v_h(T), o_h(T);
    #pragma omp parallel for
    for (size_t i = 0; i < T; ++i) {
        q_h[i] = __bfloat162float(q_bf16_h[i]);
        k_h[i] = __bfloat162float(k_bf16_h[i]);
        v_h[i] = __bfloat162float(v_bf16_h[i]);
    }
    attention_cpu(q_h.data(), k_h.data(), v_h.data(), o_h.data(), B, H, N, D);

    kernel(q_bf16_d.data().get(),
           k_bf16_d.data().get(),
           v_bf16_d.data().get(),
           o_bf16_d.data().get(),
           B, H, N, D);

    thrust::host_vector<bf16> o_bf16_h = o_bf16_d;
    thrust::host_vector<float> o_f(T);
    #pragma omp parallel for
    for (size_t i = 0; i < T; ++i) o_f[i] = __bfloat162float(o_bf16_h[i]);

    return check_output(o_f, o_h, T);
}

// int test_kernel_once(kernel_t kernel, size_t B, size_t H, size_t N, size_t D){
//     size_t T = B * H * N * D;

//     thrust::uniform_real_distribution<float> dist(-5.0, 8.0);
//     thrust::default_random_engine rng(69);

//     thrust::host_vector<float> q_h(T), k_h(T), v_h(T), o_h(T);
//     thrust::generate(q_h.begin(), q_h.end(), [&]{ return dist(rng); });
//     thrust::generate(k_h.begin(), k_h.end(), [&]{ return dist(rng); });
//     thrust::generate(v_h.begin(), v_h.end(), [&]{ return dist(rng); });
//     // thrust::generate(o_h.begin(), o_h.end(), [&]{ return dist(rng); });

//     thrust::device_vector<float> q_d = q_h, k_d = k_h, v_d = v_h, o_d = o_h;

//     attention_cpu(q_h.data(), k_h.data(), v_h.data(), o_h.data(), B, H, N, D);
//     kernel(q_d.data().get(), k_d.data().get(), v_d.data().get(), o_d.data().get(), B, H, N, D);

//     return check_output(o_d, o_h, T);
// }

void test_kernels_all(
    const std::vector<std::pair<const char*, kernel_t>>& kernels,
    const std::vector<config_t>& configs)
{
    for(const auto& cfg : configs){
        std::cout << "\n[TEST] B=" << cfg.B << " H=" << cfg.H << " N=" << cfg.N << " D=" << cfg.D << "\n";
        for(const auto& kv : kernels){
            const char* name = kv.first;
            kernel_t fn = kv.second;
            int ok = test_kernel_once(fn, cfg.B, cfg.H, cfg.N, cfg.D);
            if (!ok) std::cout << "  " << name << " : FAILED\n";
            else     std::cout << "  " << name << " : PASSED\n";
        }
    }
}

static inline void print_box(
    size_t B, size_t H, size_t N, size_t D,
    const std::vector<std::tuple<std::string, bench_result_t>>& rows)
{
    size_t w_name = 0;
    for (auto& r : rows) w_name = std::max(w_name, std::get<0>(r).size());
    w_name = std::max<size_t>(w_name, 24);

    const int w_time = 12;
    const int w_thr  = 12;

    std::ostringstream head;
    head << "B=" << B << " H=" << H << " N=" << N << " D=" << D;

    size_t inner = w_name + w_time + w_thr + 8;
    size_t box_w = std::max(inner, head.str().size()+2);

    auto line = [&](char c){ std::cout << '+' << std::string((int)box_w, c) << "+\n"; };

    line('-');
    std::cout << "| " << std::left << std::setw((int)box_w-1) << head.str() << "|\n";
    line('-');
    std::cout << "| " << std::left  << std::setw((int)w_name) << "Kernel"
              << " | " << std::right << std::setw(w_time) << "Time (ms)"
              << " | " << std::right << std::setw(w_thr)  << "GFLOP/s" << " |\n";
    line('-');

    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(3);
    for (auto& r : rows){
        const std::string& name = std::get<0>(r);
        const bench_result_t& br = std::get<1>(r);
        std::cout << "| " << std::left  << std::setw((int)w_name) << name
                  << " | " << std::right << std::setw(w_time) << br.time_ms
                  << " | " << std::right << std::setw(w_thr)  << br.gflops_per_s
                  << " |\n";
    }
    line('-');
}

void benchmark_configs(
    const std::vector<std::pair<const char*, kernel_t>>& kernels,
    const std::vector<config_t>& configs,
    int warmup, int rep)
{
    thrust::uniform_real_distribution<float> dist(-5.0, 5.0);
    thrust::default_random_engine rng(69);

    for(const auto& cfg : configs){
        size_t B = cfg.B, H = cfg.H, N = cfg.N, D = cfg.D;
        size_t T = B * H * N * D;

        thrust::host_vector<bf16> q_h(T), k_h(T), v_h(T), o_h(T);
        thrust::generate(q_h.begin(), q_h.end(), [&]{ return dist(rng); });
        thrust::generate(k_h.begin(), k_h.end(), [&]{ return dist(rng); });
        thrust::generate(v_h.begin(), v_h.end(), [&]{ return dist(rng); });
        thrust::generate(o_h.begin(), o_h.end(), [&]{ return dist(rng); });

        thrust::device_vector<bf16> q_d = q_h, k_d = k_h, v_d = v_h, o_d = o_h;

        std::vector<std::tuple<std::string, bench_result_t>> rows;
        rows.reserve(kernels.size());

        for(const auto& kv : kernels){
            const char* name = kv.first;
            kernel_t fn = kv.second;
            bench_result_t br = benchmark_ex(
                fn,
                q_d.data().get(), k_d.data().get(), v_d.data().get(), o_d.data().get(),
                B, H, N, D,
                warmup, rep,
                flops_attention
            );
            rows.emplace_back(name, br);
        }
        print_box(B, H, N, D, rows);
    }
}

int main(){
    std::vector<std::pair<const char*, kernel_t>> kernels = {
        {"attention_simple", attention_simple},
        {"attention_v2_tc", attention_v2_tc}
    };

    std::vector<config_t> configs = {
        // {4, 8, 64, 128},
        {32, 8, 64, 128},
        // {32, 8, 128, 128},
        // {32, 8, 256, 128},
    };

    test_kernels_all(kernels, configs);

    int warmup = 5;
    int rep    = 200;
    benchmark_configs(kernels, configs, warmup, rep);

    return 0;
}
