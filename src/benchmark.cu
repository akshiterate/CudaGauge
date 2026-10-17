// Benchmark engine: warm up, time repeats, validate.
//
// CPU timing uses std::chrono around cpu_work().
// GPU timing uses CUDA events around the gpu work lambda, so only GPU
// work (copies + kernel, or kernel alone) counts, never host overhead.
// One warm-up pass of each side runs untimed to exclude first-call
// costs such as lazy initialization and cold caches.

#include "cudagauge/cudagauge.h"

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

namespace cudagauge {

void cuda_check(int error_code, const char* what, const char* file, int line) {
    if (error_code != 0) {
        std::printf("CUDA error at %s:%d: %s (code %d)\n", file, line, what, error_code);
        std::fflush(stdout);
        std::exit(1);
    }
}

#define CHECK(what) cuda_check((int)cudaGetLastError(), what, __FILE__, __LINE__)

static double time_cpu_once(const std::function<void()>& work) {
    auto start = std::chrono::steady_clock::now();
    work();
    auto stop = std::chrono::steady_clock::now();
    double seconds = (double)(stop - start).count() / 1e9;
    return seconds * 1000.0;
}

static double time_gpu_once(const std::function<void()>& work) {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cuda_check((int)cudaEventCreate(&start_event), "event create", __FILE__, __LINE__);
    cuda_check((int)cudaEventCreate(&stop_event), "event create", __FILE__, __LINE__);
    cuda_check((int)cudaEventRecord(start_event), "event record", __FILE__, __LINE__);
    work();
    cuda_check((int)cudaEventRecord(stop_event), "event record", __FILE__, __LINE__);
    cuda_check((int)cudaEventSynchronize(stop_event), "event sync", __FILE__, __LINE__);
    float ms = 0.0f;
    cuda_check((int)cudaEventElapsedTime(&ms, start_event, stop_event), "elapsed", __FILE__,
               __LINE__);
    cuda_check((int)cudaEventDestroy(start_event), "event destroy", __FILE__, __LINE__);
    cuda_check((int)cudaEventDestroy(stop_event), "event destroy", __FILE__, __LINE__);
    CHECK("gpu work");
    return (double)ms;
}

Measurement run_custom_size(CustomWorkload& work,
                            long long n,
                            TimingMode mode,
                            int repeats) {
    Measurement result;
    result.size_n = n;
    result.repeats = repeats;
    result.valid = false;
    if (repeats < 1) {
        repeats = 1;
    }
    // One-time device init, kept out of every measurement.
    cuda_check((int)cudaFree(0), "device init", __FILE__, __LINE__);
    work.setup(n);
    // Warm-up passes, untimed.
    work.cpu_work();
    if (mode == TimingMode::EndToEnd) {
        work.gpu_e2e_work();
    } else {
        work.gpu_kernel_work();
    }
    CHECK("warmup");
    // Timed repeats.
    std::vector<double> cpu_samples;
    std::vector<double> gpu_samples;
    for (int r = 0; r < repeats; r++) {
        cpu_samples.push_back(time_cpu_once(work.cpu_work));
    }
    for (int r = 0; r < repeats; r++) {
        if (mode == TimingMode::EndToEnd) {
            gpu_samples.push_back(time_gpu_once(work.gpu_e2e_work));
        } else {
            gpu_samples.push_back(time_gpu_once(work.gpu_kernel_work));
        }
    }
    result.cpu_ms = median_of(cpu_samples);
    result.gpu_ms = median_of(gpu_samples);
    result.valid = work.validate();
    work.teardown();
    return result;
}

std::vector<Measurement> calibrate_custom(CustomWorkload& work,
                                           const std::vector<long long>& sizes,
                                           TimingMode mode,
                                           int repeats) {
    std::vector<Measurement> rows;
    for (size_t i = 0; i < sizes.size(); i++) {
        rows.push_back(run_custom_size(work, sizes[i], mode, repeats));
    }
    return rows;
}

}  // namespace cudagauge
