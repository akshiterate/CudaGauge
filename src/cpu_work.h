#pragma once

// Plain CPU declarations shared by workloads_cpu.cpp (definitions) and
// workloads_gpu.cu (called from cpu_work lambdas and reference checks).
// Keeping them in one tiny header avoids duplicated declarations.

namespace cudagauge {

void vecadd_cpu(const float* a, const float* b, float* out, long long n);
float reduce_cpu(const float* data, long long n);
void matmul_cpu(const float* a, const float* b, float* out, long long dim);

}  // namespace cudagauge
