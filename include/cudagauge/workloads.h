#pragma once

// Factories for the three built-in workloads. Each factory returns a
// CustomWorkload whose lambdas capture no state: all buffers live in
// file-static storage inside workloads_gpu.cu, so one workload object
// of each kind is active at a time. Use a workload, then call its
// teardown before building the next one of the same kind.
//
// Size meanings:
//   vecadd / reduce : n is the element count.
//   matmul          : n is the matrix dimension (matrices are n x n).

#include "cudagauge/cudagauge.h"

namespace cudagauge {

CustomWorkload make_vecadd_workload();
CustomWorkload make_reduce_workload();
CustomWorkload make_matmul_workload();

}  // namespace cudagauge
