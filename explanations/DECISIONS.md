# Design decisions

1. **Library first, CLI second.** All logic lives in `cudagauge.lib`;
   `main.cpp` only declares plans and prints. No benchmark logic is
   duplicated between library and CLI.
2. **One `CustomWorkload` struct, no inheritance/templates.** Smallest
   interface covering differing signatures, memory models, validation.
3. **CUDA-free public header.** `cuda_check` takes error-as-int so plain
   C++ consumers need no CUDA include path.
4. **chrono for CPU, CUDA events for GPU; warm-up + medians.** Spec
   compliance plus outlier resistance. Repeats: 5 (vector), 3 (matmul).
5. **Hand-rolled least squares.** ~20 inspectable lines, no dependency;
   equal weighting documented as a limitation.
6. **Search never touches the model.** Geometric scan (two-win stop),
   all brackets kept, 3-round bisection, fresh-measurement verification,
   honest `found=false` with reason. Budget counts every measurement.
7. **Naive CPU + naive GPU, same Release /O2.** The crossover reflects
   parallelism/transfers, not one side's cleverness. No MKL/cuBLAS.
8. **Reduction tail documented, not hidden.** Per-block partials finish
   on CPU; timed in e2e (required output), untimed tail in resident.
9. **Static per-workload buffers.** Keeps lambdas capture-free and code
   simple; costs one-live-object-per-kind (documented, tested pattern).
10. **NMake + vcvars64 build.** CMake's VS generator reports "No CUDA
    toolset found" (toolkit's VS integration not installed; installing
    it would change system config, which was not authorized). NMake
    drives nvcc directly — no system change needed.
11. **`CMAKE_CUDA_ARCHITECTURES=86`.** RTX 2050 laptop = sm_86.
12. **Calibration/validation split + steady-win rule + fixed baseline.**
    All three exist so the formula cannot be flattered or straw-manned.
13. **Level 3 third-party integration deferred.** No suitable open-source
    C++/CUDA app was available offline in this environment; the
    independent sample app (own kernel, public headers only) proves the
    adapter boundary. Documented in PROJECT_STATUS, not claimed.
14. **Search re-measures the low edge with 1 repeat during bisection.**
    Cheap anchor; adds noise at microsecond scales — accepted and logged
    (reduce-resident unverified candidate).
15. **Level 3: verbatim upstream + thin adapter.** Upstream file kept
    byte-identical and uncompiled; adapter reuses kernel/CPU/epsilon
    with 4 documented changes (deterministic fill, CPU mirrors,
    memcpy transfers after prefetch code-101 failure on WDDM,
    ceil_div arithmetic for CUDA 12.5). No library logic duplicated.
16. **Third-party smoke test in CTest.** `--quick` mode (2 sizes, no
    search, ~0.5 s) asserts build/run/validate; full calibration stays
    a manual command producing the committed CSVs.
