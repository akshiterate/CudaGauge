# `src/main.cpp` — the `cudagauge_cli` tool

## 1. Purpose
Drives the full experiment for each workload × mode: calibrate → fit
formula → validate on fresh sizes → empirical search → write three
CSVs → print the comparison. Manual `argv` parsing, no libraries.

## 2. Concepts
- **Calibration vs validation split**: the model fits 5 sizes, then is
  judged on 5 *different interleaved* sizes. Testing on training data
  would flatter the formula.
- **Steady crossover** (`observed_crossover`): first size where the GPU
  wins *twice in a row* in sorted order — same two-win rule as the
  search, so one noisy win cannot create an "observed" crossover.
- **Threshold accuracy**: fraction of validation sizes where "GPU iff
  n ≥ threshold" picks the actually faster side. Compares formula
  threshold, search threshold, and a fixed middle-size baseline fairly.

## 3–5. Walkthrough
- `Plan` struct: workload factory + calib/valid size lists + search
  range/growth + repeats. vecadd/reduce share element-count sizes;
  matmul uses dimensions (cubic cost, fewer/smaller sizes, repeats=3).
- `observed_crossover`: insertion sort by `size_n` (sizes arrive grouped
  calib-then-valid), then first double GPU win, else −1.
- `threshold_accuracy(rows, threshold)`: skips invalid rows.
- `run_plan`: times calibration with chrono (reported as calibration
  cost), fits both lines, predicts within `[first_calib, search_max]`,
  measures validation, merges for the observed value, runs search with
  budget 24, writes `*_calibration/_validation/_search.csv`, prints
  fits, prediction (+EXTRAPOLATED flag), observed, abs/rel errors,
  accuracies, calibration seconds, invalid-row count.
- `main`: parses `--workload`, `--mode`, `--out`, `--repeats`; unknown
  flags/modes exit non-zero.

## 6. Why this design
One plan table drives everything: adding a workload is one struct, and
every workload gets the identical protocol — the fairness requirement.

## 7. Example
```
build\cudagauge_cli.exe --workload matmul --mode e2e --repeats 3 --out results
```

## 8. Common mistakes
- Running Debug builds for "results" (2–10× slower CPU; use Release).
- Changing sizes between formula-fit and search runs and comparing them.

## 9. Dependencies
Public headers + library. Writes `results/*.csv`, logged in
`EXPERIMENT_LOG.md`.
