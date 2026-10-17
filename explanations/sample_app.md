# `examples/sample_app.cu` — Level 2 independent integration

## 1. Purpose
Proves the public API works for code the library has never seen: a
self-contained program using **only** `cudagauge/cudagauge.h` plus its
own kernel (`out = a·x + b`). The spec's Level 2 test.

## 2. Concepts
- Same adapter duties as built-ins: own buffers, own kernel, own
  validate — the library only drives and times.
- `cudaMemcpy` inside `gpu_e2e_work` is legitimate: transfers are the
  measured cost, not scaffolding.

## 3–5. Walkthrough
- `scale_shift_kernel`: grid-stride kernel, one `a*x[i]+b` per element.
- `app` namespace statics mirror the built-in pattern (malloc'd host,
  `cudaMalloc`'d device, unit-wide constants `kA/kB`).
- `setup` fills `h_x` with `(i%53)*0.02f` and pre-copies to device;
  `cpu_work` is a plain loop; e2e/kernel/validate follow the same
  contract as built-ins (validate uses a hand-rolled absolute/relative
  check instead of `close_enough` to show developers may write their own).
- `main`: calibrates 4 sizes, prints each row (aborts non-zero on any
  invalid), fits + predicts, searches, writes `results/sample_app.csv`.

## 6. Why this design
Copying the built-in *pattern* without including built-in *code* is the
strongest integration proof: it exercises the adapter boundary exactly
as a third party would.

## 7. Example
```
build\sample_app.exe
# prints 4 calibration rows + formula/search verdicts, writes CSV
```

## 8. Common mistakes
- Including internal headers (`cpu_work.h`, `workloads.h`) here would
  void the independence claim — don't.
- Running from a directory without `results/` (the write then fails
  loudly; the CLI has the same requirement).

## 9. Dependencies
Public header + CUDA runtime + `crossover` lib. No internal headers.
