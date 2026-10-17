# `src/workloads_cpu.cpp` + `src/cpu_work.h` — CPU reference code

## 1. Purpose
The three CPU implementations every GPU result is validated against
*and* timed against: `vecadd_cpu`, `reduce_cpu`, `matmul_cpu`.
`cpu_work.h` declares them for the GPU file; the `.cpp` defines them.

## 2. Concepts
- **Reference implementation**: the trusted version. GPU outputs must
  match it within tolerance before any timing counts.
- `double` accumulator in reduce/matmul: summing millions of floats in
  `float` loses precision; accumulating in `double` then casting back
  keeps the reference accurate. The GPU kernels do the same per-thread.

## 3–5. Walkthrough
- `vecadd_cpu(a, b, out, n)`: one loop, `out[i] = a[i] + b[i]`. O(n).
- `reduce_cpu(data, n)`: one loop summing into a `double`, returned as
  `float`. O(n).
- `matmul_cpu(a, b, out, dim)`: naive i/j/k triple loop,
  `out[row][col] = Σ a[row][k]·b[k][col]`. O(dim³). Deliberately
  unblocked: straightforward and fair (same math as the GPU kernel),
  built with the same Release `/O2` as everything else.
- `cpu_work.h`: the three declarations, nothing else — avoids
  duplicating prototypes in two files.

## 6. Why this design
Plain loops are the honest baseline: any competent developer's first
CPU version looks like this. A vectorized/MKL CPU would move every
crossover upward; the choice is documented, not hidden (DECISIONS.md).

## 7. Example
```cpp
float out[4];
vecadd_cpu(a, b, out, 4);  // out[i] = a[i] + b[i]
```

## 8. Common mistakes
- "Optimizing" only one side (e.g. `-O2` GPU vs debug CPU) — the build
  uses one Release configuration for both.
- Float accumulation order differences: CPU sums serially, GPU sums
  per-block partials — exact bitwise equality is impossible, hence
  tolerance-based validation.

## 9. Dependencies
None. Called from `src/workloads_gpu.cu` lambdas and validated in tests.
