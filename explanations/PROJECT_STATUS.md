# Project status

## Built (all working, tested on RTX 2050)
- Reusable static library `cudagauge` (public: `cudagauge.h`,
  `workloads.h`; impl: timing, benchmark, model, search, csv,
  CPU + GPU workloads).
- CLI `cudagauge_cli` (--workload/--mode/--repeats/--out).
- Level 1 tests: vecadd/reduce/matmul CPU↔GPU match, both modes, incl.
  odd sizes. Level 2: `sample_app` (own kernel, public API only) passes
  and writes `results/sample_app.csv`.
- Level 3: NVIDIA cuda-samples `vectorAdd` adapted behind the public API
  (`examples/third_party/vectorAdd/`; verbatim upstream + adapter +
  README). Builds, validates in both modes, produces 6 real CSVs
  (`results/thirdparty_*`). Resident: formula n*=113793, search verified
  163840 — matches built-in vecadd-resident within one grid step.
- Unit tests: model (exact recovery + all refusal paths), search
  (synthetic crossover bracketed + verified).
- `ctest`: **100% passed (4/4)** (incl. `thirdparty_vectorAdd --quick`).
- Real benchmark CSVs: 18 built-in + 6 third-party + sample app.
- Explanations for every source/build file + INDEX, DECISIONS,
  EXPERIMENT_LOG, this file. README with integration guide.

## Reproduce
```
cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >NUL && cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release'
cmd /c '"..." >NUL && cmake --build build'
ctest --test-dir build --output-on-failure
cmd /c build\cudagauge_cli.exe --out results
cmd /c build\sample_app.exe
build\vectorAdd_cudagauge.exe --out results
```

## Known limitations
1. One live object per built-in workload kind (static buffers).
2. Formula assumes linearity in N; wrong scale for matmul-in-dim.
3. Search abstains ("unverified") when boundary timings are
   noise-dominated (~<0.1 ms); see EXPERIMENT_LOG.
4. Reduction resident mode times kernel only; e2e includes CPU tail sum.
5. Naive CPU/GPU implementations; sm_86 binary (recompile for other GPUs).
6. `results/` CSVs are single-run medians, WDDM Windows, no clock locking.
7. Third-party adapter: CPU runs on host mirrors (not unified buffers);
   e2e transfers use `cudaMemcpy` (prefetch unsupported on this WDDM
   setup); `cuda::ceil_div` replaced (missing in CUDA 12.5). See its README.
