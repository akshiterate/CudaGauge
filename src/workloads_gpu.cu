// GPU kernels plus the three built-in workload factories.
//
// Cost definitions (kept identical on every run):
//   EndToEnd : H2D copies of inputs + kernel + D2H copy of output.
//   Resident : kernel launch + completion only. Inputs are copied to
//              the device once in setup(); the output stays on the
//              device until validate() copies it back (untimed).
//
// Reduction honesty note: the GPU kernel produces one partial sum per
// block; the final sum of those partials runs on the CPU. In EndToEnd
// mode that tail is part of the timed work (it is required output).
// In Resident mode only the kernel is timed.

#include "cudagauge/workloads.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cpu_work.h"

namespace cudagauge {

// ---- kernels (one thread block handles a tile, threads stride) ----

static __global__ void vecadd_kernel(const float* a, const float* b, float* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        out[i] = a[i] + b[i];
    }
}

static __global__ void reduce_kernel(const float* in, float* partial, long long n) {
    __shared__ float tile[256];
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;
    for (; i < n; i += (long long)gridDim.x * blockDim.x) {
        thread_sum += (double)in[i];
    }
    tile[threadIdx.x] = (float)thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            tile[threadIdx.x] += tile[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = tile[0];
    }
}

static __global__ void matmul_kernel(const float* a, const float* b, float* out, long long dim) {
    long long total = dim * dim;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; i < total; i += step) {
        long long row = i / dim;
        long long col = i % dim;
        double sum = 0.0;
        for (long long k = 0; k < dim; k++) {
            sum += (double)a[row * dim + k] * (double)b[k * dim + col];
        }
        out[i] = (float)sum;
    }
}

// ---- small helpers ----

static int num_blocks(long long n, int threads) {
    long long blocks = (n + threads - 1) / threads;
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 65535) {
        blocks = 65535;
    }
    return (int)blocks;
}

static void fill_pattern(float* data, long long n) {
    for (long long i = 0; i < n; i++) {
        data[i] = (float)((i % 97) + 1) * 0.01f;
    }
}

static bool close_enough(const float* cpu, const float* gpu, long long n, float tol) {
    for (long long i = 0; i < n; i++) {
        float allowed = tol * (1.0f + std::fabs(cpu[i]));
        if (std::fabs(cpu[i] - gpu[i]) > allowed) {
            return false;
        }
    }
    return true;
}

// ---- vector addition workload ----

namespace vecadd_state {
static float* h_a = nullptr;
static float* h_b = nullptr;
static float* h_cpu = nullptr;
static float* h_gpu = nullptr;
static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static long long cur = 0;

static void free_all() {
    std::free(h_a);
    std::free(h_b);
    std::free(h_cpu);
    std::free(h_gpu);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    h_a = h_b = h_cpu = h_gpu = nullptr;
    d_a = d_b = d_c = nullptr;
    cur = 0;
}
}  // namespace vecadd_state

CustomWorkload make_vecadd_workload() {
    using namespace vecadd_state;
    CustomWorkload w;
    w.name = "vecadd";
    w.setup = [](long long n) {
        free_all();
        cur = n;
        h_a = (float*)std::malloc((size_t)n * sizeof(float));
        h_b = (float*)std::malloc((size_t)n * sizeof(float));
        h_cpu = (float*)std::malloc((size_t)n * sizeof(float));
        h_gpu = (float*)std::malloc((size_t)n * sizeof(float));
        cuda_check((int)cudaMalloc(&d_a, (size_t)n * sizeof(float)), "vecadd alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_b, (size_t)n * sizeof(float)), "vecadd alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_c, (size_t)n * sizeof(float)), "vecadd alloc", __FILE__,
                   __LINE__);
        fill_pattern(h_a, n);
        fill_pattern(h_b, n);
        cuda_check((int)cudaMemcpy(d_a, h_a, (size_t)n * sizeof(float), cudaMemcpyHostToDevice),
                   "vecadd h2d", __FILE__, __LINE__);
        cuda_check((int)cudaMemcpy(d_b, h_b, (size_t)n * sizeof(float), cudaMemcpyHostToDevice),
                   "vecadd h2d", __FILE__, __LINE__);
    };
    w.cpu_work = []() { vecadd_cpu(h_a, h_b, h_cpu, cur); };
    w.gpu_e2e_work = []() {
        cudaMemcpy(d_a, h_a, (size_t)cur * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, (size_t)cur * sizeof(float), cudaMemcpyHostToDevice);
        vecadd_kernel<<<num_blocks(cur, 256), 256>>>(d_a, d_b, d_c, cur);
        cudaMemcpy(h_gpu, d_c, (size_t)cur * sizeof(float), cudaMemcpyDeviceToHost);
    };
    w.gpu_kernel_work = []() { vecadd_kernel<<<num_blocks(cur, 256), 256>>>(d_a, d_b, d_c, cur); };
    w.teardown = []() { free_all(); };
    w.validate = []() {
        cuda_check((int)cudaMemcpy(h_gpu, d_c, (size_t)cur * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                   "vecadd d2h", __FILE__, __LINE__);
        return close_enough(h_cpu, h_gpu, cur, 1e-4f);
    };
    return w;
}

// ---- reduction workload ----

namespace reduce_state {
static float* h_in = nullptr;
static float* h_partial = nullptr;
static float* d_in = nullptr;
static float* d_partial = nullptr;
static float cpu_sum = 0.0f;
static long long cur = 0;
static int blocks = 0;

static void free_all() {
    std::free(h_in);
    std::free(h_partial);
    if (d_in) cudaFree(d_in);
    if (d_partial) cudaFree(d_partial);
    h_in = h_partial = nullptr;
    d_in = d_partial = nullptr;
    cur = 0;
    blocks = 0;
}

static float finish_partials() {
    cuda_check((int)cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(float),
                               cudaMemcpyDeviceToHost),
               "reduce d2h", __FILE__, __LINE__);
    double total = 0.0;
    for (int i = 0; i < blocks; i++) {
        total += (double)h_partial[i];
    }
    return (float)total;
}
}  // namespace reduce_state

CustomWorkload make_reduce_workload() {
    using namespace reduce_state;
    CustomWorkload w;
    w.name = "reduce";
    w.setup = [](long long n) {
        free_all();
        cur = n;
        blocks = num_blocks(n, 256);
        h_in = (float*)std::malloc((size_t)n * sizeof(float));
        h_partial = (float*)std::malloc((size_t)blocks * sizeof(float));
        cuda_check((int)cudaMalloc(&d_in, (size_t)n * sizeof(float)), "reduce alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_partial, (size_t)blocks * sizeof(float)), "reduce alloc",
                   __FILE__, __LINE__);
        fill_pattern(h_in, n);
        cuda_check((int)cudaMemcpy(d_in, h_in, (size_t)n * sizeof(float), cudaMemcpyHostToDevice),
                   "reduce h2d", __FILE__, __LINE__);
    };
    w.cpu_work = []() { cpu_sum = reduce_cpu(h_in, cur); };
    w.gpu_e2e_work = []() {
        cudaMemcpy(d_in, h_in, (size_t)cur * sizeof(float), cudaMemcpyHostToDevice);
        reduce_kernel<<<blocks, 256>>>(d_in, d_partial, cur);
        // Tail sum of per-block partials (required to produce the answer).
        cudaMemcpy(h_partial, d_partial, (size_t)blocks * sizeof(float),
                   cudaMemcpyDeviceToHost);
        double total = 0.0;
        for (int i = 0; i < blocks; i++) {
            total += (double)h_partial[i];
        }
        h_partial[0] = (float)total;
    };
    w.gpu_kernel_work = []() { reduce_kernel<<<blocks, 256>>>(d_in, d_partial, cur); };
    w.teardown = []() { free_all(); };
    w.validate = []() {
        float gpu_sum = finish_partials();
        float allowed = 1e-3f * (1.0f + std::fabs(cpu_sum));
        return std::fabs(cpu_sum - gpu_sum) <= allowed;
    };
    return w;
}

// ---- matrix multiplication workload (n = dimension) ----

namespace matmul_state {
static float* h_a = nullptr;
static float* h_b = nullptr;
static float* h_cpu = nullptr;
static float* h_gpu = nullptr;
static float* d_a = nullptr;
static float* d_b = nullptr;
static float* d_c = nullptr;
static long long dim = 0;

static void free_all() {
    std::free(h_a);
    std::free(h_b);
    std::free(h_cpu);
    std::free(h_gpu);
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    h_a = h_b = h_cpu = h_gpu = nullptr;
    d_a = d_b = d_c = nullptr;
    dim = 0;
}
}  // namespace matmul_state

CustomWorkload make_matmul_workload() {
    using namespace matmul_state;
    CustomWorkload w;
    w.name = "matmul";
    w.setup = [](long long n) {
        free_all();
        dim = n;
        long long count = n * n;
        h_a = (float*)std::malloc((size_t)count * sizeof(float));
        h_b = (float*)std::malloc((size_t)count * sizeof(float));
        h_cpu = (float*)std::malloc((size_t)count * sizeof(float));
        h_gpu = (float*)std::malloc((size_t)count * sizeof(float));
        cuda_check((int)cudaMalloc(&d_a, (size_t)count * sizeof(float)), "matmul alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_b, (size_t)count * sizeof(float)), "matmul alloc", __FILE__,
                   __LINE__);
        cuda_check((int)cudaMalloc(&d_c, (size_t)count * sizeof(float)), "matmul alloc", __FILE__,
                   __LINE__);
        fill_pattern(h_a, count);
        fill_pattern(h_b, count);
        cuda_check((int)cudaMemcpy(d_a, h_a, (size_t)count * sizeof(float),
                                   cudaMemcpyHostToDevice),
                   "matmul h2d", __FILE__, __LINE__);
        cuda_check((int)cudaMemcpy(d_b, h_b, (size_t)count * sizeof(float),
                                   cudaMemcpyHostToDevice),
                   "matmul h2d", __FILE__, __LINE__);
    };
    w.cpu_work = []() { matmul_cpu(h_a, h_b, h_cpu, dim); };
    w.gpu_e2e_work = []() {
        long long count = dim * dim;
        cudaMemcpy(d_a, h_a, (size_t)count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, (size_t)count * sizeof(float), cudaMemcpyHostToDevice);
        matmul_kernel<<<num_blocks(count, 256), 256>>>(d_a, d_b, d_c, dim);
        cudaMemcpy(h_gpu, d_c, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
    };
    w.gpu_kernel_work = []() {
        long long count = dim * dim;
        matmul_kernel<<<num_blocks(count, 256), 256>>>(d_a, d_b, d_c, dim);
    };
    w.teardown = []() { free_all(); };
    w.validate = []() {
        long long count = dim * dim;
        cuda_check((int)cudaMemcpy(h_gpu, d_c, (size_t)count * sizeof(float),
                                   cudaMemcpyDeviceToHost),
                   "matmul d2h", __FILE__, __LINE__);
        return close_enough(h_cpu, h_gpu, count, 1e-3f);
    };
    return w;
}

}  // namespace cudagauge
