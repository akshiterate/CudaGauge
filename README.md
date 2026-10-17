# CudaGauge

Reusable C++/CUDA library + CLI that finds the workload size where your GPU implementation becomes faster than your CPU implementation — by formula, by measurement, and honestly compared.

Tested on: RTX 2050 laptop 4 GB (sm_86), CUDA 12.5, MSVC 2022, Windows.

---

## What problem it solves

You already have a CPU implementation and a CUDA GPU implementation of the same workload. Somewhere between small and large inputs the GPU wins (or never wins). Picking that switch point by guesswork is risky: too early and you pay transfer/kernel-launch overhead for nothing; too late and you leave performance on the table.

CudaGauge answers two questions:

1. **Is there a crossover at all?** — Does the GPU ever beat the CPU, for this workload, in this mode, on this machine?
2. **Where is it?** — What workload size `n*` should you use as your dispatch threshold?

It does this two independent ways and compares them:

| Method | How it works | Best for |
|--------|--------------|----------|
| **Formula-based prediction** | Fit a line `time = intercept + slope * n` to a few calibration measurements, solve `cpu(n) == gpu(n)` for `n`. | When time is linear in `n` (e.g. streaming kernels). Fast — only a handful of points. |
| **Adaptive empirical search** | Geometric sampling across sizes, bisects the first bracket where the winner flips, verifies with fresh repeats. | When the signal beats noise but the formula breaks (cubic cost, flat overhead, extrapolation). |

The CLI runs both under an identical calibration/validation split protocol — the formula is never graded on the data that fitted it — and scores both against a fixed middle-size baseline.

### What "N" and "the crossover" mean here

**One number means different things per workload:**

| Workload | `n` counts | Calibration range | Formula input |
|----------|-----------|-------------------|---------------|
| `vecadd` | elements | 4096 – 4 194 304 elements | element count (time ∝ n) |
| `reduce` | elements | 4096 – 4 194 304 elements | element count (time ∝ n) |
| `matmul` | **matrix dimension** (`n×n` matrix, `n³` ops) | 64 – 2048 (dim) | dimension as if time ∝ n — structurally wrong (real cost ∝ n³), kept to demonstrate the failure mode |

**Three different "crossover" definitions are used—don't mix them up:**

1. **Formula `n*`** — where the two *fitted straight lines* intersect. Exact only if cost really is linear in `n` (see above).
2. **Observed `n`** — the first *scanned grid point* where the GPU wins twice in a row. Grid = geometric (`n, 4n, 16n, …`), so this is quantized: the true transition lies *somewhere between the neighboring grid points*, typically anywhere in a 4× band. Reported errors (below) are distances to that quantized point, not exact distances to the true crossover.
3. **Search `n*`** — the refined upper edge of the first bracket, re-measured on fresh repeat batches. Interior candidates are labeled `verified-interior`; candidates verified at `max_n` are `boundary-candidate` — the true crossover may lie *at or beyond* the window edge, which is a finding about the window, not a universally established crossover.

**Error formula:** `abs_err = |formula_n* − observed_n|`, `rel_err = abs_err / observed_n` (denominator = the observed grid point). Both inherit the grid quantization above.

---

## Build

```bat
:: 1. Enter the MSVC x64 environment
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL'

:: 2. Configure (fresh build directory)
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release

:: 3. Build
cmake --build build

:: 4. Run the test suite
ctest --test-dir build --output-on-failure
```

- **Generator note:** CMake's Visual Studio generator reports "No CUDA toolset" on machines without the toolkit's VS integration; NMake + `vcvars64.bat` works without any system change.
- **Other GPUs:** add `-DCMAKE_CUDA_ARCHITECTURES=<yours>` (e.g. `80` for A100, `86` for RTX 20xx/30xx laptop).
- Targets produced: `cudagauge.lib`, `cudagauge_cli.exe`, `sample_app.exe`, `vectorAdd_cudagauge.exe`, plus three test executables.

---

## Run

```bat
:: All workloads × both modes, writes CSVs to a fresh results directory
:: (refuses to overwrite existing files unless --overwrite is passed)
build\cudagauge_cli.exe --out results_2026-10-18

:: Single workload / mode / repeat override
build\cudagauge_cli.exe --workload matmul --mode e2e --repeats 3 --out results_2026-10-18

:: Extra one-off measurements at given sizes (no fit, no search) —
:: useful to probe near a suspected boundary
build\cudagauge_cli.exe --workload reduce --mode e2e --out results_probe --probe 12582912,16777216,25165824

:: Level 2: independent client example (only public headers).
:: Refuses to overwrite results/sample_app.csv; archive or delete it first.
build\sample_app.exe

:: Level 3: NVIDIA cuda-samples vectorAdd behind the public adapter
build\vectorAdd_cudagauge.exe --out results_vectoradd
build\vectorAdd_cudagauge.exe --quick --overwrite --out build\test_results   :: ctest smoke
```

Modes: `e2e` (end-to-end, includes H2D + kernel + D2H), `resident` (kernel only, data already on device), or `both`.

Workloads: `vecadd`, `reduce`, `matmul`, or `all`.

Every run writes `<workload>_<mode>_{calibration,validation,search}.csv` (plus `_probe.csv` with `--probe`) into `--out`. Recorded measurements are never silently replaced: the CLI pre-checks every planned file and refuses before benchmarking anything.

---

## What I discovered: formula vs. manual search

> **Provenance note:** the table below is from the **single 2026-10-15 run**; its raw CSVs were overwritten during a rebuild and only the console log survives (`results/2026-10-15_historical/console_log.txt`). Later runs are kept in their own dated directories under `results/` (see `results/MANIFEST.md`) and are never merged into this table. The core finding—but **neither method wins universally**—replicated on every run.

| Workload | Mode | Formula `n*` | Observed `n` | Formula error | Search `n*` | Search verified? |
|----------|------|--------------|--------------|---------------|-------------|------------------|
| vecadd | e2e | none | none in range | — | none | GPU never wins |
| vecadd | resident | 85304 | 262144 | 68% | 163840 | ✓ |
| reduce | e2e | none | none ≤ 2M | — | 16M (edge) | boundary candidate in this run; **not reproduced** in two later runs (GPU loses ≤ 25M) |
| reduce | resident | 32852 | 32768 | 0.3% | unverified | noise-dominated |
| matmul | e2e | 92 | 48 | 92% | 64 | ✓ |
| matmul | resident | 91 | 48 | 90% | unverified | noise-dominated |

### Key lessons

**1. The formula is precise when linearity holds — and misleading when it doesn't.**
- `reduce-resident` fits a line almost perfectly (0.3% error). Streaming kernel time really is linear in `N` there.
- `matmul` costs `∝ dim³` but was fitted as `∝ dim`. Formula predicted `n* ≈ 91`; observed crossover was `48` (90% error). The model was structurally the wrong shape.

**2. Flat overhead breaks the formula.**
- `vecadd-resident` GPU time is launch-overhead dominated, not linear. Extrapolating from small-N medians missed the true crossover by 68%.

**3. The empirical search is honest about uncertainty.**
- Where noise swamps the signal (`reduce-resident`, `matmul-resident`), the search abstained rather than inventing a number. Refined searches landed on ~0.04–0.09 ms differences — below measurement noise.
- Verification now uses **up to two fresh repeat batches**, and *all* must agree the GPU wins; a single disagreeing batch marks the candidate `unverified`. That makes the search *more* resistant to timing noise, but noise cannot be eliminated — it can only require more agreement.

**4. Transfer-inclusive (e2e) often means the GPU never wins.**
- `vecadd-e2e`: three full memory passes per `N` cannot beat one CPU pass; both formula and search correctly reported *absence* of a crossover.
- `reduce-e2e`: the 2026-10-15 run logged a boundary candidate at the 16M edge; two later runs (2026-10-17 full re-run, 2026-10-18 re-check with extra near-boundary probes at 4M–25M) show the GPU *losing* everywhere in that range. On this evidence there is **no established reduce-e2e crossover ≤ 25M** — the edge result was a single-run, possibly noise-driven observation. See `results/2026-10-18_reduce_e2e_recheck/`.

**5. A threshold estimator should check linearity first.**
Use the formula only after confirming the cost is linear in `n`; otherwise fall back to the adaptive search. The library never rewrites your dispatch logic — it only recommends a size.

---

## Integrate into your app (5 steps)

1. **Include** `include/cudagauge/cudagauge.h`, link `cudagauge.lib`.
2. **Fill a `CustomWorkload`**: `setup(n)` allocates/fills (untimed); `cpu_work` / `gpu_e2e_work` / `gpu_kernel_work` each do one timed pass; `validate()` compares outputs; `teardown()` frees.
3. **Calibrate**: `calibrate_custom(w, sizes, mode, repeats)` → fit with `fit_line`, predict with `predict_crossover`.
4. **Cross-check**: `adaptive_search` — trust measurement over formula when they disagree.
5. **Dispatch**: use the recommended size only in *your* dispatcher; the library never silently modifies production logic.

See `examples/sample_app.cu` for a complete external-client example, and `examples/third_party/vectorAdd/` for integrating third-party code (NVIDIA's `vectorAdd`) through the same adapter.

---

## Repository layout

```
CudaGauge/
├── CMakeLists.txt          # build: cudagauge lib, cudagauge_cli, examples, tests
├── README.md               # this file
├── include/cudagauge/
│   ├── cudagauge.h         # public library API (CUDA-free, plain C++ includable)
│   └── workloads.h         # built-in workload factories
├── src/
│   ├── timing.cpp          # median_of, mode_to_string
│   ├── benchmark.cu        # run_custom_size, calibrate_custom
│   ├── model.cpp           # fit_line, predict_crossover (formula contestant)
│   ├── search.cpp          # adaptive_search (empirical contestant)
│   ├── csv.cpp             # write_measurements_csv
│   ├── workloads_cpu.cpp   # vecadd/reduce/matmul reference loops
│   ├── workloads_gpu.cu    # kernels + factories + validation
│   ├── main.cpp            # cudagauge_cli: plans, split, comparison, CSVs
│   └── cpu_work.h          # internal CPU helper declarations
├── examples/
│   ├── sample_app.cu       # Level 2: independent client using only public headers
│   └── third_party/vectorAdd/  # Level 3: NVIDIA vectorAdd behind the adapter
├── tests/
│   ├── test_correctness.cpp   # Level 1: CPU↔GPU match, both modes, odd sizes
│   ├── test_model.cpp         # formula unit tests (exact fit, parallels, extrapolation, invalid-row filtering)
│   └── test_search.cpp        # search tests: synthetic crossover, no-bracket, boundary, multi-bracket, noisy, budget
├── results/                  # benchmark output; each run in its own dated
│   │                         # subdirectory (see results/MANIFEST.md for
│   │                         # provenance of every surviving artifact)
│   ├── MANIFEST.md           # what each results/* directory is
│   ├── 2026-10-15_historical/  # console log of the original run
│   ├── 2026-10-17_regen_run/   # full CSVs of the regeneration run
│   └── 2026-10-18_reduce_e2e_recheck/  # unstable reduce-e2e re-evaluation
```

---

## Reproducing results

```bat
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake --build build'
ctest --test-dir build --output-on-failure
build\cudagauge_cli.exe --out results_reproduce
```

Expect ±20% jitter on small-N medians (clocks, thermals, background CPU); crossovers may shift one grid step. **Run-to-run disagreement at a boundary is a real finding — record it (each run in its own directory) instead of re-running until the number you hoped for appears.** Never compare Debug-build numbers with Release results — use Release for both CPU and GPU sides.
