// Straightforward CPU reference implementations. They are deliberately
// plain loops: fast enough to be fair, simple enough to learn from.
// Built with the same Release /O2 flags as everything else so the
// CPU/GPU comparison is honest (see DECISIONS.md).

#include "cpu_work.h"

namespace cudagauge {

void vecadd_cpu(const float* a, const float* b, float* out, long long n) {
    for (long long i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

float reduce_cpu(const float* data, long long n) {
    double total = 0.0;
    for (long long i = 0; i < n; i++) {
        total += (double)data[i];
    }
    return (float)total;
}

void matmul_cpu(const float* a, const float* b, float* out, long long dim) {
    for (long long row = 0; row < dim; row++) {
        for (long long col = 0; col < dim; col++) {
            double sum = 0.0;
            for (long long k = 0; k < dim; k++) {
                sum += (double)a[row * dim + k] * (double)b[k * dim + col];
            }
            out[row * dim + col] = (float)sum;
        }
    }
}

}  // namespace cudagauge
