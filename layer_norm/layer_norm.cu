#include <cooperative_groups.h>
#include <iostream>
#include <kernels.h>
#include <omp.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <utils.h>

void layer_norm_cpu(float *X, float *Y, float *gamma, float *beta, size_t B,
                    size_t F, size_t D1, size_t D2) {
  size_t stride = F * D1 * D2;
  float *batch_mean = new float[B];
  float *batch_std = new float[B];

#pragma omp parallel for
  for (int batch = 0; batch < B; batch++) {
    float mean = 0.0f;
    float var = 0.0f;
    float *X_batch = X + (batch * stride);
    for (int it = 0; it < stride; it++) {
      mean += X_batch[it];
      var += X_batch[it] * X_batch[it];
    }
    mean /= (float)stride;
    batch_mean[batch] = mean;
    batch_std[batch] = sqrtf((var / (float)stride) - (mean * mean) + 1e-5f);
  }
#pragma omp parallel for
  for (int batch = 0; batch < B; batch++) {
    float mean = batch_mean[batch];
    float std = batch_std[batch];
    float *X_batch = X + (batch * stride);
    float *Y_batch = Y + (batch * stride);
    for (int it = 0; it < stride; it++) {
      Y_batch[it] = ((X_batch[it] - mean) / std) * gamma[it] + beta[it];
    }
  }
}

int check_output(thrust::device_vector<float> &data1,
                 thrust::host_vector<float> &data2, size_t n) {
  thrust::host_vector<float> data1_cpu = data1;
  for (int i = 0; i < n; i++) {
    float diff = fabs(data1_cpu[i] - data2[i]);
    if (diff > 0.02) {
      std::cout << "CPU : " << data2[i] << " GPU : " << data1_cpu[i] << "\n";
      return 0;
    }
  }
  return 1;
}

#include <iomanip>
#include <string>

typedef void (*kernel_t)(float *, float *, float *, float *, size_t, size_t,
                         size_t, size_t);

struct config_t {
  size_t B, F, D1, D2;
};

struct bench_result_t {
  float time_ms;       // avg ms per run
  double flops;        // total FLOPs per run
  double gflops_per_s; // throughput
};

static inline double flops_layer_norm(size_t B, size_t F, size_t D1,
                                      size_t D2) {
  double s = (double)F * (double)D1 * (double)D2;
  return (double)B * (s * 7.0 + 6.0);
}

// Single-run benchmark that returns avg time + FLOPs + GFLOP/s
bench_result_t benchmark_ex(kernel_t kernel, float *X, float *Y, float *gamma,
                            float *beta, size_t B, size_t F, size_t D1,
                            size_t D2, int warmup, int rep,
                            double (*flops_fn)(size_t, size_t, size_t,
                                               size_t) = flops_layer_norm) {
  for (int it = 0; it < warmup; it++) {
    kernel(X, Y, gamma, beta, B, F, D1, D2);
  }
  CUDAtimer *timer = new CUDAtimer();
  timer->start_timer();
  for (int it = 0; it < rep; it++) {
    kernel(X, Y, gamma, beta, B, F, D1, D2);
  }
  timer->stop_timer();
  float avg_ms = timer->get_time() / (float)rep;
  delete timer;

  double flops = flops_fn(B, F, D1, D2);
  double gflops_per_s = (flops / (avg_ms / 1000.0)) / 1e9;
  return {avg_ms, flops, gflops_per_s};
}

// correctness test for a single kernel + single shape
int test_kernel_once(kernel_t kernel, size_t B, size_t F, size_t D1,
                     size_t D2) {
  size_t N = B * F * D1 * D2;
  size_t stride_b = F * D1 * D2;

  thrust::uniform_real_distribution<float> dist(-5.0, 8.0);
  thrust::default_random_engine rng(69);

  thrust::host_vector<float> x_h(N);
  thrust::host_vector<float> y_h(N);
  thrust::host_vector<float> gamma_h(stride_b);
  thrust::host_vector<float> beta_h(stride_b);
  thrust::generate(x_h.begin(), x_h.end(), [&] { return dist(rng); });
  thrust::generate(y_h.begin(), y_h.end(), [&] { return dist(rng); });
  thrust::generate(gamma_h.begin(), gamma_h.end(), [&] { return dist(rng); });
  thrust::generate(beta_h.begin(), beta_h.end(), [&] { return dist(rng); });

  thrust::device_vector<float> x_d = x_h;
  thrust::device_vector<float> y_d = y_h;
  thrust::device_vector<float> gamma_d = gamma_h;
  thrust::device_vector<float> beta_d = beta_h;

  // reference on CPU
  layer_norm_cpu(x_h.data(), y_h.data(), gamma_h.data(), beta_h.data(), B, F,
                 D1, D2);
  // float mx(-1.0f);
  // for(float data : y_h){
  //     mx = std::max<float>(mx, data);
  // }
  // std::cout << "CPU Max : " << mx << "\n";

  // DUT on GPU
  kernel(x_d.data().get(), y_d.data().get(), gamma_d.data().get(),
         beta_d.data().get(), B, F, D1, D2);

  // compare
  return check_output(y_d, y_h, N);
}

// wrapper: run correctness across many shapes for many kernels
void test_kernels_all(
    const std::vector<std::pair<const char *, kernel_t>> &kernels,
    const std::vector<config_t> &configs) {
  for (const auto &cfg : configs) {
    std::cout << "\n[TEST] B=" << cfg.B << " F=" << cfg.F << " D1=" << cfg.D1
              << " D2=" << cfg.D2 << "\n";
    for (const auto &kv : kernels) {
      const char *name = kv.first;
      kernel_t fn = kv.second;
      int ok = test_kernel_once(fn, cfg.B, cfg.F, cfg.D1, cfg.D2);
      if (!ok) {
        std::cout << "  " << name << " : FAILED\n";
      } else {
        std::cout << "  " << name << " : PASSED\n";
      }
    }
  }
}

// pretty box printer for a single config and multiple kernels
static inline void
print_box(size_t B, size_t F, size_t D1, size_t D2,
          const std::vector<std::tuple<std::string, bench_result_t>> &rows) {
  size_t w_name = 0;
  for (auto &r : rows)
    w_name = std::max(w_name, std::get<0>(r).size());
  w_name = std::max<size_t>(w_name, 24);

  const int w_time = 12;
  const int w_flop = 16;
  const int w_thr = 12;

  std::ostringstream head;
  head << "B=" << B << " F=" << F << " D1=" << D1 << " D2=" << D2;

  size_t inner = w_name + w_time + w_thr + 8;
  size_t box_w = std::max(inner, head.str().size() + 2);

  auto line = [&](char c) {
    std::cout << '+' << std::string((int)box_w, c) << "+\n";
  };

  line('-');
  std::cout << "| " << std::left << std::setw((int)box_w - 1) << head.str()
            << "|\n";
  line('-');
  std::cout << "| " << std::left << std::setw((int)w_name) << "Kernel"
            << " | " << std::right << std::setw(w_time) << "Time (ms)"
            << " | " << std::right << std::setw(w_thr) << "GFLOP/s" << " |\n";
  line('-');

  std::cout.setf(std::ios::fixed);
  std::cout << std::setprecision(3);
  for (auto &r : rows) {
    const std::string &name = std::get<0>(r);
    const bench_result_t &br = std::get<1>(r);
    std::cout << "| " << std::left << std::setw((int)w_name) << name << " | "
              << std::right << std::setw(w_time) << br.time_ms << " | "
              << std::right << std::setw(w_thr) << std::setprecision(3)
              << br.gflops_per_s << " |\n";
  }
  line('-');
}

// wrapper: benchmark an array of configs for multiple kernels and print pretty
// boxes
void benchmark_configs(
    const std::vector<std::pair<const char *, kernel_t>> &kernels,
    const std::vector<config_t> &configs, int warmup, int rep) {
  thrust::uniform_real_distribution<float> dist(-5.0, 5.0);
  thrust::default_random_engine rng(69);

  for (const auto &cfg : configs) {
    size_t B = cfg.B, F = cfg.F, D1 = cfg.D1, D2 = cfg.D2;
    size_t N = B * F * D1 * D2;
    size_t stride_b = F * D1 * D2;

    // allocate once per config
    thrust::host_vector<float> x_h(N);
    thrust::host_vector<float> y_h(N);
    thrust::host_vector<float> gamma_h(stride_b);
    thrust::host_vector<float> beta_h(stride_b);
    thrust::generate(x_h.begin(), x_h.end(), [&] { return dist(rng); });
    thrust::generate(y_h.begin(), y_h.end(), [&] { return dist(rng); });
    thrust::generate(gamma_h.begin(), gamma_h.end(), [&] { return dist(rng); });
    thrust::generate(beta_h.begin(), beta_h.end(), [&] { return dist(rng); });

    thrust::device_vector<float> x_d = x_h;
    thrust::device_vector<float> y_d = y_h;
    thrust::device_vector<float> gamma_d = gamma_h;
    thrust::device_vector<float> beta_d = beta_h;

    std::vector<std::tuple<std::string, bench_result_t>> rows;
    rows.reserve(kernels.size());

    for (const auto &kv : kernels) {
      const char *name = kv.first;
      kernel_t fn = kv.second;
      bench_result_t br = benchmark_ex(
          fn, x_d.data().get(), y_d.data().get(), gamma_d.data().get(),
          beta_d.data().get(), B, F, D1, D2, warmup, rep, flops_layer_norm);
      rows.emplace_back(name, br);
    }
    print_box(B, F, D1, D2, rows);
  }
}

int main() {
  std::vector<std::pair<const char *, kernel_t>> kernels = {
      {"layer_norm_simple_dsmem", layer_norm_simple_dsmem},
      // {"layer_norm_simple_async", layer_norm_simple_async},
      // {"layer_norm_simple", layer_norm_simple},
      {"layer_norm_simple_dsmem_async_v2", layer_norm_simple_dsmem_async_v2},
      // {"layer_norm_simple_dsmem_async_v1", layer_norm_simple_dsmem_async_v1},
      // {"layer_norm_simple_persistant", layer_norm_simple_persistant}
  };

  std::vector<config_t> configs = {{4, 128, 64, 64}, {4, 4, 4, 4}};

  test_kernels_all(kernels, configs);

  int warmup = 5;
  int rep = 1000;
  benchmark_configs(kernels, configs, warmup, rep);

  return 0;
}
