# `src/benchmark.cu` — warm up, time, validate

## 1. Purpose
Implements `run_custom_size` and `calibrate_custom`: the only place
that ever starts the clock. `.cu` extension because GPU timing needs
CUDA events.

## 2. Concepts
- **`std::chrono::steady_clock`**: monotonic CPU wall clock, immune to
  system-time adjustments. Difference of two time points = elapsed time.
- **CUDA events**: GPU timestamps recorded in the GPU's own command
  stream. `cudaEventRecord(start)` → run work → `cudaEventRecord(stop)`
  → `cudaEventSynchronize(stop)` → `cudaEventElapsedTime` gives
  milliseconds of *completed GPU work only*, excluding host overhead.
- **Warm-up**: the first CUDA call initializes the driver and the first
  kernel launch moves code to the device. One untimed pass each side
  keeps these one-time costs out of every median.

## 3–5. Walkthrough
- `cuda_check(int code, const char* what, const char* file, int line)`:
  prints location and exits on any CUDA error. Never silently continue
  with a broken device state.
- `CHECK` macro: feeds `cudaGetLastError()` plus file/line after each
  GPU phase (warm-up and every timed repeat batch share it via the
  helpers).
- `time_cpu_once(work)`: one chrono bracket around `work()`, seconds
  converted to milliseconds.
- `time_gpu_once(work)`: creates two events, records start, runs `work`,
  records stop, synchronizes (waits until the GPU actually finished),
  reads elapsed ms, destroys events, checks errors.
- `run_custom_size(work, n, mode, repeats)`:
  1. `cudaFree(0)` forces one-time device init outside all timings.
  2. `setup(n)`, then one untimed `cpu_work()` + one untimed GPU pass.
  3. `repeats` timed CPU passes → median; `repeats` timed GPU passes
     (e2e or kernel-only per `mode`) → median.
  4. `validate()` once; `teardown()`. Returns the `Measurement`.
- `calibrate_custom(...)`: loops over sizes, fresh setup/teardown per
  size so no state leaks between sizes.

## 6. Why this design
Chrono for CPU and events for GPU is exactly what the spec demands;
measuring the developer's lambdas (not internal buffers) means custom
workloads get identical treatment to built-ins.

## 7. Example
```cpp
CustomWorkload w = make_reduce_workload();
Measurement m = run_custom_size(w, 262144, TimingMode::EndToEnd, 5);
// m.cpu_ms / m.gpu_ms are medians of 5; m.valid says outputs matched.
```

## 8. Common mistakes
- Forgetting `cudaEventSynchronize` (measures launch time, ~µs, not work).
- Comparing e2e numbers against resident numbers (different definitions).
- `repeats < 1` (clamped to 1 inside, but medians of 1 are just samples).

## 9. Dependencies
Public header + CUDA runtime. Called by `src/search.cpp`,
`src/main.cpp`, tests, and the sample app.
