# `examples/third_party/vectorAdd/` — Level 3 external integration

## 1. Purpose
Proves CudaGauge works on code it was never designed for: NVIDIA's
open-source `vectorAdd` sample, adapted through the public
`CustomWorkload` API without touching the library. Three files: the
verbatim upstream copy, the adapter, and a README.

## 2. Concepts
- **Unified memory** (`cudaMallocManaged`): one pointer valid on host
  and device; the driver migrates pages on demand. No explicit copies
  in upstream — which forces the adapter to define what "end-to-end"
  means (see below).
- **Prefetch vs copy**: `cudaMemPrefetchAsync` (async hint) failed here
  with code 101 on WDDM, so synchronous `cudaMemcpy` moves the same
  bytes. Same cost model, portable call.
- **`int` vs `long long`**: upstream counts elements in `int`; the
  adapter guards `n <= INT_MAX` and casts once.

## 3–5. Walkthrough
- `vectorAdd_upstream.cu`: byte-identical upstream, never compiled.
- `vectorAdd_cudagauge.cu`:
  - License header kept; kernel `upstream_vecAdd` and
    `upstream_serialVecAdd` copied unchanged (renamed with prefix to
    avoid confusion, identical bodies).
  - `grid_blocks()`: `(n+255)/256`, value-identical to upstream's
    `cuda::ceil_div(n, 256)` (header missing in CUDA 12.5).
  - `setup`: managed buffers (GPU path, faithful) + malloc mirrors
    (CPU path, identical values, deterministic fill). No prefetch.
  - `cpu_work`: upstream serial function on mirrors.
  - `gpu_e2e_work`: H2D copies → kernel → sync → D2H copy.
  - `gpu_kernel_work`: kernel → sync (warm-up pass migrates pages in).
  - `validate`: D2H copy + upstream-style `1e-5` comparison.
  - `main`: `--quick` (2 sizes, 2 repeats, no search; used by ctest) or
    full run (5 calib + 5 valid sizes, both modes, search budget 24,
    CSVs to `results/thirdparty_*.csv`). Non-zero exit on any invalid.
- `README.md`: provenance, callback table, commands, numbers, limits.

## 6. Why this design
Verbatim-file-plus-adapter keeps "what upstream does" reviewable
against "what we changed" (4 items, all listed in the README). CPU
mirrors trade pointer-identity for measurement hygiene; the epsilon
check proves computational equivalence.

## 7. Example
```
build\vectorAdd_cudagauge.exe --out results
# [resident] formula: has=1 n*=113793 ... search: found=1 n*=163840
```

## 8. Common mistakes
- Running CPU work on the managed buffers (migrates pages every repeat;
  inflates CPU, evicts GPU residency).
- Assuming e2e ≈ resident after warm-up (true only without per-repeat
  transfers — the adapter's copies prevent that).
- Compiling the upstream file into the same binary (duplicate `main`).

## 9. Dependencies
Public header + CUDA runtime + `crossover` lib. Registered in
`CMakeLists.txt`; smoke test in CTest. Results feed EXPERIMENT_LOG.
