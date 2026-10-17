// Level 2 test: an independent developer-style application.
//
// It uses ONLY the public headers (cudagauge/cudagauge.h) plus its own
// kernel. It registers a saxpy-like workload (out = a*x + b), runs
// calibration, fits the formula, runs the empirical search, and
// exports its own CSV. If this builds and runs, the public adapter
// works for code the library has never seen.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cudagauge/cudagauge.h"

using namespace cudagauge;

static __global__ void scale_shift_kernel(const float* x, float* out, float a, float b,
                                          long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) {
        out[i] = a * x[i] + b;
    }
}

namespace app {
static float* h_x = nullptr;
static float* h_cpu = nullptr;
static float* h_gpu = nullptr;
static float* d_x = nullptr;
static float* d_out = nullptr;
static long long cur = 0;
static const float kA = 2.5f;
static const float kB = -1.0f;

static void free_all() {
    std::free(h_x);
    std::free(h_cpu);
    std::free(h_gpu);
    if (d_x) cudaFree(d_x);
    if (d_out) cudaFree(d_out);
    h_x = h_cpu = h_gpu = nullptr;
    d_x = d_out = nullptr;
    cur = 0;
}
}  // namespace app

int main() {
    using namespace app;
    CustomWorkload work;
    work.name = "scale_shift";
    work.setup = [](long long n) {
        free_all();
        cur = n;
        h_x = (float*)std::malloc((size_t)n * sizeof(float));
        h_cpu = (float*)std::malloc((size_t)n * sizeof(float));
        h_gpu = (float*)std::malloc((size_t)n * sizeof(float));
        cuda_check((int)cudaMalloc(&d_x, (size_t)n * sizeof(float)), "app alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_out, (size_t)n * sizeof(float)), "app alloc", __FILE__,
                   __LINE__);
        for (long long i = 0; i < n; i++) {
            h_x[i] = (float)(i % 53) * 0.02f;
        }
        cuda_check((int)cudaMemcpy(d_x, h_x, (size_t)n * sizeof(float),
                                   cudaMemcpyHostToDevice),
                   "app h2d", __FILE__, __LINE__);
    };
    work.cpu_work = []() {
        for (long long i = 0; i < cur; i++) {
            h_cpu[i] = kA * h_x[i] + kB;
        }
    };
    work.gpu_e2e_work = []() {
        cudaMemcpy(d_x, h_x, (size_t)cur * sizeof(float), cudaMemcpyHostToDevice);
        long long blocks = (cur + 255) / 256;
        if (blocks > 65535) blocks = 65535;
        scale_shift_kernel<<<(int)blocks, 256>>>(d_x, d_out, kA, kB, cur);
        cudaMemcpy(h_gpu, d_out, (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost);
    };
    work.gpu_kernel_work = []() {
        long long blocks = (cur + 255) / 256;
        if (blocks > 65535) blocks = 65535;
        scale_shift_kernel<<<(int)blocks, 256>>>(d_x, d_out, kA, kB, cur);
    };
    work.teardown = []() { free_all(); };
    work.validate = []() {
        cuda_check((int)cudaMemcpy(h_gpu, d_out, (size_t)cur * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                   "app d2h", __FILE__, __LINE__);
        for (long long i = 0; i < cur; i++) {
            float allowed = 1e-4f * (1.0f + h_cpu[i] * (h_cpu[i] > 0 ? 1 : -1));
            if (allowed < 0) allowed = -allowed;
            float diff = h_cpu[i] - h_gpu[i];
            if (diff < 0) diff = -diff;
            if (diff > allowed) return false;
        }
        return true;
    };

    std::vector<long long> sizes;
    sizes.push_back(4096);
    sizes.push_back(65536);
    sizes.push_back(1048576);
    sizes.push_back(4194304);
    std::vector<Measurement> rows = calibrate_custom(work, sizes, TimingMode::EndToEnd, 5);
    for (size_t i = 0; i < rows.size(); i++) {
        std::printf("n=%lld cpu=%.4f ms gpu=%.4f ms valid=%d\n", rows[i].size_n, rows[i].cpu_ms,
                    rows[i].gpu_ms, (int)rows[i].valid);
        if (!rows[i].valid) {
            std::printf("SAMPLE APP FAILED validation\n");
            return 1;
        }
    }
    std::vector<long long> ns;
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;
    for (size_t i = 0; i < rows.size(); i++) {
        ns.push_back(rows[i].size_n);
        cpu_times.push_back(rows[i].cpu_ms);
        gpu_times.push_back(rows[i].gpu_ms);
    }
    Prediction p = predict_crossover(fit_line(ns, cpu_times), fit_line(ns, gpu_times), 4096,
                                     16777216);
    std::printf("formula: has=%d n*=%lld %s\n", (int)p.has_crossover, p.crossover_n,
                p.note.c_str());
    SearchResult s = adaptive_search(work, TimingMode::EndToEnd, 1024, 16777216, 4.0, 5, 20);
    std::printf("search: found=%d n*=%lld (%s)\n", (int)s.found, s.crossover_n, s.note.c_str());
    if (!write_measurements_csv("results/sample_app.csv", "scale_shift", "e2e", rows)) {
        std::printf("could not write results/sample_app.csv\n");
        return 1;
    }
    std::printf("sample app OK\n");
    return 0;
}
