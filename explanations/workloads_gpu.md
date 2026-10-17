# `src/workloads_gpu.cu` — kernels and workload factories

## 1. Purpose
The three CUDA kernels plus the `make_*` factories that bundle CPU
functions, kernels, buffers, and validation into `CustomWorkload`
objects. The largest file; each workload follows the same pattern.

## 2. Concepts
- **Kernel** (`__global__` function): runs on the GPU; launched with
  `<<<blocks, threads>>>`. Here 256 threads/block, grid-stride loops
  (`for (; i < n; i += gridSize)`) so any size works with ≤ 65535 blocks.
- **`__shared__` + `__syncthreads()`** (reduce): threads in a block
  cooperate through fast on-chip memory; the halve-and-add loop reduces
  256 values to 1 per block.
- **H2D / D2H**: `cudaMemcpy` host↔device transfers; they *are* the e2e
  cost for small sizes.
- **File-static state**: buffers live in `vecadd_state` / `reduce_state`
  / `matmul_state` namespaces so the lambdas (which capture nothing)
  can reach them. Trade-off: one active object per kind (documented in
  `workloads_h.md`).

## 3–5. Walkthrough
- `num_blocks(n, threads)`: ceil division clamped to [1, 65535].
- `fill_pattern`: deterministic inputs `(i%97+1)*0.01f` — small values,
  no overflow in matmul sums, reproducible across runs.
- `close_enough`: per-element relative tolerance check.
- **vecadd**: `setup` mallocs host + device, fills, pre-copies inputs
  (for resident mode). `gpu_e2e_work` = H2D + kernel + D2H.
  `gpu_kernel_work` = kernel only. `validate` copies the device result
  back (untimed) and compares with 1e-4 tolerance.
- **reduce**: kernel writes one partial per block; the tail sum of
  partials runs on the CPU. In e2e mode that tail is *inside* the timed
  region (it is required to produce the answer); in resident mode only
  the kernel is timed and `validate` finishes the sum untimed. Honesty
  note is in the file header comment.
- **matmul**: one thread per output element, inner `k` loop in `double`.
  Same cost definitions as vecadd.
- Each `teardown` frees host (`std::free`) and device (`cudaFree`) and
  nulls pointers so double-free is impossible.

## 6. Why this design
Naive-but-correct kernels mirror the naive CPU: the crossover reflects
parallelism and transfers, not one side's cleverness. Grid-stride loops
avoid a different kernel config per size.

## 7. Example
```cpp
vecadd_kernel<<<num_blocks(n,256), 256>>>(d_a, d_b, d_c, n);
// d_c[i] = d_a[i] + d_b[i] for all i < n
```

## 8. Common mistakes
- Reading `h_gpu` after a resident run without the validate copy (stale).
- Forgetting e2e and resident timings have different definitions.
- Assuming two same-kind workloads can coexist (static buffers).

## 9. Dependencies
Public `workloads.h`, internal `cpu_work.h`, CUDA runtime, CPU file.
Used by CLI and correctness test.
