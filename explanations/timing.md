# `src/timing.cpp` — median and mode names (no CUDA here)

## 1. Purpose
Implements the two public helpers that need no GPU: `median_of` and
`mode_to_string`. Kept in a plain `.cpp` file so the logic stays
testable without a device.

## 2. Concepts
- **Median vs mean**: five repeats like `[1.0, 1.1, 1.0, 9.5, 1.0]`
  (one OS hiccup) average to 2.7 but have median 1.0. Benchmarks use
  medians so background activity cannot fake a result.
- `std::sort` from `<algorithm>` orders the copy.

## 3–5. Walkthrough
- `median_of(std::vector<double> values)`: takes the vector **by value**
  (a copy), so sorting never disturbs the caller's data. Empty input
  returns 0.0. Odd count returns the middle element; even count averages
  the two middle elements.
- `mode_to_string(TimingMode)`: returns `"e2e"` or `"resident"` for CSV
  columns and log lines. One place defines the spelling, so files and
  console output can never disagree.

## 6. Why this design
Pass-by-value looks wasteful but the vectors hold ≤ dozens of doubles;
the copy buys freedom from aliasing bugs and reads clearly.

## 7. Example
```cpp
std::vector<double> samples = {2.0, 1.0, 9.0};
double m = crossover::median_of(samples);  // 2.0; samples unchanged
```

## 8. Common mistakes
- Using the mean of repeats (fragile to outliers).
- Timing with `rand()`-scale jitter and only 1 repeat (median of one
  sample is just the sample — increase repeats instead).

## 9. Dependencies
Only the public header. Used by `src/benchmark.cu` for every median.
