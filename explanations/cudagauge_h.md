# `include/cudagauge/cudagauge.h` — the public library API

## 1. Purpose
Single header an external developer includes to use CudaGauge. It declares
the workload adapter (`CustomWorkload`), the result structs, and every
library function. It is the contract: anything here is stable, everything
else is internal.

## 2. Concepts
- **Adapter pattern**: instead of the library calling your functions
  directly (impossible — every app differs), you fill a struct of
  `std::function` callbacks and the library drives them.
- **`std::function<void(long long)>`**: a callable object taking a size.
  A lambda like `[](long long n) { ... }` converts to one implicitly.
- **Timing modes** (`TimingMode::EndToEnd` vs `Resident`): which GPU
  costs count. See `explanations/benchmark.md`.
- **Median**: middle value of sorted repeats; immune to single outliers.

## 3–5. Walkthrough
- `TimingMode` enum: two named choices, no magic ints at call sites.
- `Measurement`: one row of results. `valid=false` means "do not trust
  these times" (allocation or correctness failed).
- `CustomWorkload`: six callbacks plus a name.
  - `setup(n)` / `teardown()`: untimed allocate/fill/free.
  - `cpu_work`, `gpu_e2e_work`, `gpu_kernel_work`: one timed pass each.
  - `validate()`: untimed CPU-vs-GPU comparison, returns bool.
- `LinearFit` (`intercept`, `slope`, `ok`, `note`): the line
  `time = intercept + slope * n`. `ok=false` when fitting was impossible.
- `Prediction`: `has_crossover` + `crossover_n`, with `extrapolated=true`
  when the answer lies outside measured sizes.
- `SearchInterval` / `SearchResult`: the empirical answer. `history`
  keeps every measurement; `transitions` keeps every sign change so
  non-monotonic behavior is visible, not hidden.
- Function declarations: `median_of`, `mode_to_string`, `cuda_check`,
  `run_custom_size`, `calibrate_custom`, `fit_line`,
  `predict_crossover`, `adaptive_search`, `write_measurements_csv`.
- `cuda_check(int code, ...)`: takes the CUDA error as `int` so this
  header compiles without CUDA includes. Call sites cast `(int)err`.

## 6. Why this design
One struct with plain callbacks is the smallest interface that still
handles different CPU/GPU signatures and memory models. No templates,
no base classes, no macros — a developer learns it in five minutes.

## 7. Example
```cpp
#include "cudagauge/cudagauge.h"
cudagauge::CustomWorkload w;
w.setup = [](long long n) { /* allocate n elements */ };
w.cpu_work = []() { /* one CPU pass */ };
// ... fill the rest, then:
cudagauge::Measurement m =
    cudagauge::run_custom_size(w, 65536, cudagauge::TimingMode::EndToEnd, 5);
```

## 8. Common mistakes
- Timing allocation inside `cpu_work` instead of `setup` (inflates CPU).
- Forgetting `validate` (fast wrong answers look like a GPU win).
- Treating `Prediction.crossover_n` as truth when `extrapolated=true`.

## 9. Dependencies
No dependencies. Implemented across `src/timing.cpp`, `src/benchmark.cu`,
`src/model.cpp`, `src/search.cpp`, `src/csv.cpp`.
