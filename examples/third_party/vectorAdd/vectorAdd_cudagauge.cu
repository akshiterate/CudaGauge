// Level 3 integration: NVIDIA cuda-samples `vectorAdd` under CudaGauge.
//
// Upstream: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/vectorAdd/vectorAdd.cu
// Verbatim copy (not compiled): vectorAdd_upstream.cu in this directory.
//
// What is reused verbatim from upstream: the vecAdd kernel, serialVecAdd,
// the cuda::ceil_div launch calculation, cudaMallocManaged buffers, and
// the epsilon comparison idea. The NVIDIA copyright header above is kept.
// Intentional changes (see README.md): deterministic fill instead of
// rand()/srand(time); separate plain-malloc CPU mirrors so page migration
// never pollutes CPU timing; long-long sizes with an int-range guard.
//
// Unified-memory cost definitions (see README.md for why copies, not
// prefetches, model the transfers):
//   EndToEnd : cudaMemcpy mirrors->managed (H2D) + kernel + sync +
//              managed->host copy (D2H), every repeat.
//   Resident : kernel + sync only. The untimed warm-up pass migrates
//              pages to the device; the CPU path never touches the
//              unified buffers, so repeats measure resident work.
// Only the public header cudagauge/cudagauge.h is used. No CudaGauge
// benchmarking, model, or search logic is duplicated here.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

// NOTE: upstream sizes its grid with cuda::ceil_div from <cuda/cmath>,
// which only ships with newer CUDA toolkits (absent in CUDA 12.5).
// The equivalent integer arithmetic below computes the identical value.

#include "cudagauge/cudagauge.h"

using namespace cudagauge;

// ---- upstream computations, unchanged ----

static __global__ void upstream_vecAdd(float* A, float* B, float* C, int vectorLength) {
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

static void upstream_serialVecAdd(float* A, float* B, float* C, int length) {
    for (int i = 0; i < length; i++) {
        C[i] = A[i] + B[i];
    }
}

// ---- adapter state ----

namespace tp {
static float* u_a = nullptr;  // unified buffers (GPU path, as upstream)
static float* u_b = nullptr;
static float* u_c = nullptr;
static float* h_a = nullptr;  // plain host mirrors (CPU path)
static float* h_b = nullptr;
static float* h_cpu = nullptr;
static float* h_gpu = nullptr;
static long long cur = 0;

static void free_all() {
    if (u_a) cudaFree(u_a);
    if (u_b) cudaFree(u_b);
    if (u_c) cudaFree(u_c);
    std::free(h_a);
    std::free(h_b);
    std::free(h_cpu);
    std::free(h_gpu);
    u_a = u_b = u_c = nullptr;
    h_a = h_b = h_cpu = h_gpu = nullptr;
    cur = 0;
}

static int grid_blocks(long long n) {
    // Identical to upstream's cuda::ceil_div((int)n, 256).
    long long blocks = (n + 255) / 256;
    if (blocks < 1) {
        return 1;
    }
    return (int)blocks;
}
}  // namespace tp

static CustomWorkload make_upstream_vectoradd() {
    using namespace tp;
    CustomWorkload w;
    w.name = "thirdparty_vectorAdd";
    w.setup = [](long long n) {
        free_all();
        if (n > INT_MAX) {
            std::printf("size %lld exceeds upstream int range\n", n);
            std::exit(1);
        }
        cur = n;
        cuda_check((int)cudaMallocManaged(&u_a, (size_t)n * sizeof(float)), "managed alloc",
                   __FILE__, __LINE__);
        cuda_check((int)cudaMallocManaged(&u_b, (size_t)n * sizeof(float)), "managed alloc",
                   __FILE__, __LINE__);
        cuda_check((int)cudaMallocManaged(&u_c, (size_t)n * sizeof(float)), "managed alloc",
                   __FILE__, __LINE__);
        h_a = (float*)std::malloc((size_t)n * sizeof(float));
        h_b = (float*)std::malloc((size_t)n * sizeof(float));
        h_cpu = (float*)std::malloc((size_t)n * sizeof(float));
        h_gpu = (float*)std::malloc((size_t)n * sizeof(float));
        for (long long i = 0; i < n; i++) {
            float v = (float)((i % 97) + 1) * 0.01f;
            h_a[i] = v;
            h_b[i] = v;
            u_a[i] = v;
            u_b[i] = v;
        }
    };
    w.cpu_work = []() { upstream_serialVecAdd(h_a, h_b, h_cpu, (int)cur); };
    w.gpu_e2e_work = []() {
        size_t bytes = (size_t)cur * sizeof(float);
        cudaMemcpy(u_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(u_b, h_b, bytes, cudaMemcpyHostToDevice);
        upstream_vecAdd<<<grid_blocks(cur), 256>>>(u_a, u_b, u_c, (int)cur);
        cudaDeviceSynchronize();
        cudaMemcpy(h_gpu, u_c, bytes, cudaMemcpyDeviceToHost);
    };
    w.gpu_kernel_work = []() {
        upstream_vecAdd<<<grid_blocks(cur), 256>>>(u_a, u_b, u_c, (int)cur);
        cudaDeviceSynchronize();
    };
    w.teardown = []() { free_all(); };
    w.validate = []() {
        size_t bytes = (size_t)cur * sizeof(float);
        cuda_check((int)cudaMemcpy(h_gpu, u_c, bytes, cudaMemcpyDeviceToHost), "copy d2h",
                   __FILE__, __LINE__);
        for (long long i = 0; i < cur; i++) {
            float diff = h_cpu[i] - h_gpu[i];
            if (diff < 0) {
                diff = -diff;
            }
            if (diff > 0.00001f) {
                return false;
            }
        }
        return true;
    };
    return w;
}

static int run_mode(CustomWorkload& work, TimingMode mode, const std::vector<long long>& calib,
                    const std::vector<long long>& valid, int repeats, const std::string& out_dir,
                    bool full) {
    std::string mode_name = mode_to_string(mode);
    std::vector<Measurement> calib_rows = calibrate_custom(work, calib, mode, repeats);
    for (size_t i = 0; i < calib_rows.size(); i++) {
        std::printf("[%s] n=%lld cpu=%.4f ms gpu=%.4f ms valid=%d\n", mode_name.c_str(),
                    calib_rows[i].size_n, calib_rows[i].cpu_ms, calib_rows[i].gpu_ms,
                    (int)calib_rows[i].valid);
        if (!calib_rows[i].valid) {
            std::printf("THIRD-PARTY FAILED validation\n");
            return 1;
        }
    }
    std::vector<long long> ns;
    std::vector<double> cpu_times;
    std::vector<double> gpu_times;
    for (size_t i = 0; i < calib_rows.size(); i++) {
        ns.push_back(calib_rows[i].size_n);
        cpu_times.push_back(calib_rows[i].cpu_ms);
        gpu_times.push_back(calib_rows[i].gpu_ms);
    }
    Prediction p =
        predict_crossover(fit_line(ns, cpu_times), fit_line(ns, gpu_times), calib.front(), calib.back());
    std::printf("[%s] formula: has=%d n*=%lld %s\n", mode_name.c_str(), (int)p.has_crossover,
                p.crossover_n, p.note.c_str());
    std::string base = out_dir + "/thirdparty_vectorAdd_" + mode_name;
    write_measurements_csv(base + "_calibration.csv", "thirdparty_vectorAdd", mode_name,
                           calib_rows);
    if (full) {
        std::vector<Measurement> valid_rows = calibrate_custom(work, valid, mode, repeats);
        write_measurements_csv(base + "_validation.csv", "thirdparty_vectorAdd", mode_name,
                               valid_rows);
        SearchResult s = adaptive_search(work, mode, 1024, 16777216, 4.0, repeats, 24);
        write_measurements_csv(base + "_search.csv", "thirdparty_vectorAdd", mode_name,
                               s.history);
        std::printf("[%s] search: found=%d n*=%lld (%s)\n", mode_name.c_str(), (int)s.found,
                    s.crossover_n, s.note.c_str());
    }
    return 0;
}

int main(int argc, char** argv) {
    bool quick = false;
    std::string out_dir = "results";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else {
            std::printf("unknown arg: %s\n", arg.c_str());
            return 1;
        }
    }
    CustomWorkload work = make_upstream_vectoradd();
    std::vector<long long> calib;
    std::vector<long long> valid;
    int repeats = 5;
    if (quick) {
        calib.push_back(4096);
        calib.push_back(65536);
        repeats = 2;
    } else {
        calib.push_back(4096);
        calib.push_back(16384);
        calib.push_back(65536);
        calib.push_back(262144);
        calib.push_back(1048576);
        valid.push_back(8192);
        valid.push_back(32768);
        valid.push_back(131072);
        valid.push_back(524288);
        valid.push_back(2097152);
    }
    if (run_mode(work, TimingMode::EndToEnd, calib, valid, repeats, out_dir, !quick)) {
        return 1;
    }
    if (run_mode(work, TimingMode::Resident, calib, valid, repeats, out_dir, !quick)) {
        return 1;
    }
    std::printf("third-party vectorAdd OK\n");
    return 0;
}
