# CudaGauge — final study guide

## Architecture in one paragraph
A developer describes their CPU and GPU implementations as six callbacks
in a `CustomWorkload`. The benchmark engine (`benchmark.cu`) warms up,
times repeats with chrono/CUDA-events, and validates outputs. The same
`Measurement` rows feed two independent contestants: the linear formula
(`model.cpp`) and the geometric search (`search.cpp`). The CLI
(`main.cpp`) runs both under an identical calibration/validation-split
protocol, exports CSVs, and scores both against a fixed baseline.

## The math, honestly
`T=a+bN` least squares has a closed form (see INDEX); `N*=(c−a)/(b−d)`.
It works when time is linear in the size variable (reduce-resident:
0.3% error) and fails when it is not (matmul-in-dim is cubic: 90%
error) or when overhead dominates (vecadd-resident: 68% error). The
project's experimental conclusion: check linearity first; otherwise
search empirically.

## CUDA in this project
Host vs device memory and why copies dominate small N; kernel launches
and grid-stride loops; events as GPU-only stopwatches (synchronize or
you time nothing); `__shared__` block reduction; error checking after
every API call; `cudaFree(0)` init outside timings; arch sm_86.

## Testing methodology
Correctness before timing (odd sizes included); synthetic-crossover
search test with ms-scale signal; calibration/validation split so the
formula never grades its own homework; steady-double-win rule;
threshold-accuracy scoring vs a fixed baseline; medians; warm-up;
single Release configuration for both sides.

## Limitations checklist
Static one-object buffers; linearity assumption; noise abstention;
reduction tail accounting; naive implementations; sm_86-only binary;
single-run WDDM medians; Level 3 integration deferred. Each is
documented where it bites, not just here.
