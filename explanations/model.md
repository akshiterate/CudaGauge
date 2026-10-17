# `src/model.cpp` — the linear formula predictor

## 1. Purpose
Implements `fit_line` (least-squares fit of `time = intercept + slope*n`)
and `predict_crossover` (solve `cpu(n) == gpu(n)`). This is the
"formula" contestant; `src/search.cpp` is the empirical one.

## 2. Concepts
- **Ordinary least squares**: the line minimizing the sum of squared
  vertical errors. Closed form with sums:
  `slope = (m·Sxy − Sx·Sy) / (m·Sxx − Sx·Sx)`,
  `intercept = (Sy − slope·Sx) / m`.
- **Crossover algebra**: `a + b·n = c + d·n` gives `n* = (c−a)/(b−d)`.
- **Assumption (often wrong)**: time is linear in `n`. True-ish for
  streaming kernels (vecadd), false for matmul in dimension `n`
  (cost ∝ n³). The experiment log shows the formula failing exactly
  where linearity breaks.

## 3–5. Walkthrough
- `fit_line(sizes, times)`: rejects fewer than 2 points or mismatched
  lengths (`ok=false` + note). Accumulates the four sums in one loop,
  rejects `denom == 0` (all sizes identical), else applies the formulas.
  All doubles — `long long` sizes are cast once per point.
- `predict_crossover(cpu_fit, gpu_fit, min_n, max_n)`: rejects unfitted
  inputs; rejects `|b−d| < 1e-15` (parallel lines); rejects negative
  crossings; sets `extrapolated=true` when `n*` falls outside the
  measured `[min_n, max_n]` range so callers can distrust it loudly.

## 6. Why this design
Hand-rolled least squares in ~20 lines needs no library and every step
is inspectable — the point of the project. No regularization, no
weighting: documented simplicity over robustness (see DECISIONS.md).

## 7. Example
```cpp
LinearFit cpu{1.0, 2.0, true, ""}, gpu{5.0, 1.0, true, ""};
Prediction p = predict_crossover(cpu, gpu, 1, 100);
// p.has_crossover == true, p.crossover_n == 4
```

## 8. Common mistakes
- Fitting on sizes spanning 3 orders of magnitude with plain LS: the
  largest points dominate; log-scale or piecewise fits would be fairer.
- Forgetting the model was fitted on *medians* — noisy small-n points
  get equal weight.
- Using the prediction when `extrapolated=true` without saying so.

## 9. Dependencies
Public header + `<cmath>` only. Tested by `tests/test_model.cpp`,
used by `src/main.cpp` and `examples/sample_app.cu`.
