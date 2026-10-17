# `include/crossover/workloads.h` — built-in workload factories

## 1. Purpose
Declares `make_vecadd_workload()`, `make_reduce_workload()`, and
`make_matmul_workload()`. Each returns a ready-to-benchmark
`CustomWorkload`. The CLI and the correctness test use these; external
developers may also use them as reference adapters.

## 2. Concepts
- **Factory function**: a function whose job is to build and return an
  object, so callers never touch the buffers inside.
- Size meaning differs per workload (elements vs matrix dimension);
  the comment at the top of the file states the convention.

## 3–5. Walkthrough
Three declarations, one per workload, each returning `CustomWorkload`
by value. No parameters: sizes are supplied later via `setup(n)`.
One documented limitation: the lambdas use file-static storage (see
`explanations/workloads_gpu.md`), so only one live object per workload
kind is supported — use it, then let `teardown()` run before making
another.

## 6. Why this design
New workloads are added by writing one more `make_*` function. The
search, model, and CSV code never changes, satisfying the requirement
that algorithms stay workload-agnostic.

## 7. Example
```cpp
#include "crossover/workloads.h"
crossover::CustomWorkload w = crossover::make_vecadd_workload();
auto rows = crossover::calibrate_custom(w, {1024, 4096},
                                        crossover::TimingMode::Resident, 5);
```

## 8. Common mistakes
- Assuming `n` means elements for matmul (it means dimension; total
  elements are `n*n`, and cost grows as `n^3`).
- Holding two `make_vecadd_workload()` objects alive at once (shared
  static buffers — the second `setup` frees the first's data).

## 9. Dependencies
Includes `crossover/crossover.h`. Implemented in
`src/workloads_gpu.cu` (GPU side + factories) with CPU help from
`src/workloads_cpu.cpp`.
