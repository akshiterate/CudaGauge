# CudaGauge — learning index and study guide entry point

## Recommended study order
1. `cudagauge_h.md` — the contract everything else serves.
2. `timing.md` → `benchmark.md` — how any number in `results/` was born.
3. `workloads_cpu.md` → `workloads_gpu.md` → `workloads_h.md` — what runs.
4. `model.md` → `search.md` — the two competing methods.
5. `main.md` → `csv.md` — the experiment harness and its output.
6. `sample_app.md` → `tests.md` → `build.md` — integration, tests, build.
7. `third_party_vectorAdd.md` — external code behind the same adapter.
8. `DECISIONS.md` → `EXPERIMENT_LOG.md` → `PROJECT_STATUS.md` — why, what
   happened, what is left.

## How the files fit together
```
include/cudagauge/cudagauge.h   <-- developer's view (CustomWorkload + API)
include/cudagauge/workloads.h   <-- built-in factories
src/timing.cpp       median_of, mode_to_string
src/benchmark.cu     run_custom_size, calibrate_custom (clocks live here)
src/model.cpp        fit_line, predict_crossover      (formula contestant)
src/search.cpp       adaptive_search                  (empirical contestant)
src/csv.cpp          write_measurements_csv
src/workloads_cpu.cpp  vecadd/reduce/matmul reference loops
src/workloads_gpu.cu   kernels + factories + validation
src/main.cpp         cudagauge_cli: plans, split, comparison, CSVs
tests/               correctness / model / search executables
examples/sample_app.cu  independent developer-style client
examples/third_party/vectorAdd/  NVIDIA vectorAdd: verbatim upstream +
                        crossover adapter + README (Level 3)
results/             generated CSVs (never hand-edited)
```

## Glossary
- **Crossover (n\*)**: workload size where the GPU becomes faster than
  the CPU. Not a universal constant — depends on workload, mode, GPU,
  clocks, and build flags.
- **End-to-end (e2e)**: GPU time including H2D copies, kernel, D2H copy.
- **Resident**: GPU kernel time only; data already on device.
- **Calibration / validation**: sizes that fit the model vs fresh sizes
  that judge it. Never the same sizes.
- **Median of repeats**: reported time; robust to one-off OS noise.
- **Warm-up**: untimed first pass excluding init/cold-cache costs.
- **Bracket**: adjacent sizes with different winners; contains a crossover.
- **Extrapolation**: prediction outside measured sizes; flagged, not hidden.
- **Grid-stride loop**: kernel pattern handling any `n` with a fixed grid.
- **H2D / D2H**: host-to-device / device-to-host copies.
- **Least squares**: line minimizing squared errors (see equations below).
- **Steady win**: GPU faster twice in a row; filters single noisy wins.

## All equations and assumptions
- Model: `T_CPU(N) = a + b·N`, `T_GPU(N) = c + d·N`.
- Fit: `slope = (m·Sxy − Sx·Sy)/(m·Sxx − Sx·Sx)`,
  `intercept = (Sy − slope·Sx)/m` over `m` calibration medians.
- Crossover: `N* = (c − a)/(b − d)`, valid only if denominator ≠ 0,
  `N* ≥ 0`, and — the big assumption — time is actually linear in `N`.
  For matmul in dimension `n`, cost ∝ n³, so the assumption breaks and
  the formula errs (rel. error 0.90 on this machine).

## CUDA concepts used
Kernels + `<<<blocks, threads>>>` launch, grid-stride loops,
`cudaMemcpy` H2D/D2H, `cudaMalloc/cudaFree`, CUDA events for timing
(record/synchronize/elapsed), `__shared__` + `__syncthreads` block
reduction, `cudaGetLastError` checking, `cudaFree(0)` device init,
compute capability / arch `sm_86`.

## Check yourself
1. Why must validation sizes differ from calibration sizes?
2. What does Resident mode exclude that EndToEnd includes, and where in
   `workloads_gpu.cu` is each defined?
3. Why median and not mean of repeats?
4. Derive `N* = (c−a)/(b−d)` from the two model lines.
5. When does `predict_crossover` refuse, and what does each refusal mean?
6. Why does the search require two consecutive GPU wins to stop scanning?
7. Why did the formula succeed on reduce-resident (rel. err 0.003) but
   fail on matmul (0.90)? What assumption differs?
8. At ~0.05 ms timings the search refinement coin-flips: why?
9. What is unverifiable about a `Prediction` with `extrapolated=true`?
10. A developer's GPU kernel needs inputs in a special layout: which
    `CustomWorkload` callbacks handle that, and which stay untouched?
