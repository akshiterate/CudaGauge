# Level 3 integration: NVIDIA cuda-samples `vectorAdd`

## Upstream project
- Source: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/vectorAdd/vectorAdd.cu
- License: NVIDIA BSD-style (copyright header preserved verbatim in both
  files in this directory).
- What it is: element-wise `C = A + B` using **unified memory**
  (`cudaMallocManaged`), with a serial CPU version for comparison and an
  epsilon (`1e-5`) result check.
- `vectorAdd_upstream.cu` is a byte-identical copy of the upstream file
  at integration time. It is **not compiled** and **not modified**; it
  exists so reviewers can diff adapter against original.

## Adapter callbacks (`vectorAdd_cudagauge.cu`)
| Callback | Implementation |
|---|---|
| `setup(n)` | `cudaMallocManaged` A/B/C (as upstream) + plain-malloc CPU mirrors; deterministic fill; guards `n <= INT_MAX` (upstream uses `int`) |
| `cpu_work` | `serialVecAdd` on the host mirrors (upstream function, unchanged) |
| `gpu_e2e_work` | H2D copies + upstream `vecAdd` kernel + sync + D2H copy |
| `gpu_kernel_work` | upstream `vecAdd` kernel + sync only |
| `validate` | D2H copy, compare vs CPU output with upstream's `1e-5` epsilon |
| `teardown` | `cudaFree` + `free`, pointers nulled |

Only `cudagauge/cudagauge.h` is included. No library logic duplicated.

## Build / run (existing NMake + vcvars64 environment)
```
cmd /c '"...\vcvars64.bat" >NUL && cmake --build build --target vectorAdd_cudagauge'
build\vectorAdd_cudagauge.exe --out results            :: full run
build\vectorAdd_cudagauge.exe --quick --out results    :: smoke test (ctest)
```

## Actual results (RTX 2050, 2026-10-16, `results/thirdparty_*.csv`)
- **e2e**: formula predicts no crossover; search finds none; all rows
  valid. Transfer-inclusive GPU never beats the CPU to 2M elements —
  same verdict as built-in vecadd-e2e.
- **resident**: formula n\* = 113793 (in range); search verified 163840.
  Built-in vecadd-resident gave 85304 / 163840 — independent code,
  same answer within one grid step.
- Validation CSVs above show the raw medians (e.g. resident n=2097152:
  cpu 3.40 ms vs gpu 0.79 ms).

## Limitations
1. `cuda::ceil_div` from `<cuda/cmath>` (upstream master) does not exist
   in CUDA 12.5; replaced with identical `(n+255)/256` arithmetic.
2. `initArray`'s `rand()/srand(time)` replaced with deterministic fill —
   reproducible sizes, no per-size reseed cost.
3. The CPU runs on host mirrors, not the unified buffers: running the
   CPU on managed memory would migrate pages on every repeat and
   pollute both timings. Computations are identical; validated equal.
4. e2e transfers use `cudaMemcpy` because `cudaMemPrefetchAsync` returns
   `cudaErrorInvalidDevice` (code 101) on this WDDM setup. Same bytes
   moved; documented, not hidden.
5. Upstream's `int` length caps sizes at 2^31−1 (guarded; our max is 16M).
6. As with built-ins, the binary targets sm_86.
