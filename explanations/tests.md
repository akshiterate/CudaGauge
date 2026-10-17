# `tests/` + integration smoke test — tests, no framework

## 1. Purpose
`test_correctness.cpp` (Level 1: built-ins match CPU↔GPU in both modes),
`test_model.cpp` (formula unit tests), `test_search.cpp` (search on a
synthetic workload), plus the `thirdparty_vectorAdd --quick` CTest entry
(Level 3 smoke: builds, validates both modes, ~0.5 s). Plain `main()` +
`printf` + return codes — no gtest dependency, per the small-dependency
rule. Registered with CTest.

## 2. Concepts
- **Correctness before timing**: a fast wrong answer is worthless, so
  Level 1 asserts `valid==true` at normal and odd sizes (odd sizes catch
  off-by-one grid math: 100003 elements, 37×37 matmul).
- **Synthetic workload** (search test): CPU busy-loop ∝ `5000·n`, GPU
  ∝ `50M + 50·n`, so a real crossover exists near n≈10100 and the search
  must bracket-and-verify it. `volatile` accumulator stops the compiler
  deleting the loop. Lesson learned during development: microsecond
  workloads made verification a coin flip — the synthetic timings run
  tens of ms so the signal beats the noise.

## 3–5. Walkthrough
- `test_correctness.cpp`: `check_workload(factory, n, label)` runs
  `run_custom_size(..., 2 repeats)` in both modes, prints times, counts
  failures, exits 1 on any `!valid`.
- `test_model.cpp`: exact line `y=1+2x` must recover slope 2 / intercept
  1; crossing of `1+2x`/`5+x` must be 4 in-range; parallel lines → none;
  single point → `ok=false`; out-of-range crossing → `extrapolated`.
- `test_search.cpp`: builds the fake adapter, runs `adaptive_search`
  (budget 16), requires `found` and a real `[lower, upper]` bracket.

## 6. Why this design
Three executables, each testing one layer, each readable in one sitting.
`ctest --output-on-failure` runs them all.

## 7. Example
```
ctest --test-dir build --output-on-failure
# 100% tests passed, 4 tests passed out of 4
```

## 8. Common mistakes
- Search test slowness (~45 s): the burn loops are the signal; shrinking
  them reintroduces the noise flakiness documented above.
- GPU-less machines: correctness/search tests need a real device.

## 9. Dependencies
Public headers + library. Build targets in `CMakeLists.txt`.
