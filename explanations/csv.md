# `src/csv.cpp` — result export

## 1. Purpose
Implements `write_measurements_csv`: dumps any measurement vector to a
CSV file with the fixed header
`workload,mode,n,cpu_ms,gpu_ms,repeats,valid`.

## 2. Concepts
- **CSV**: comma-separated values, one row per line; opens in any
  spreadsheet or Python/pandas for plotting.
- `std::ofstream`: file output stream; `if (!out)` detects open failure.

## 3–5. Walkthrough
Opens the path (truncating), writes the header, then one line per
`Measurement`: workload/mode names supplied by the caller, `size_n`,
both medians, repeat count, and `valid` as 1/0 (numeric so filters and
plots treat it as data). Returns false if the file could not be opened
or the final write failed.

## 6. Why this design
One shared writer guarantees calibration, validation, and search files
all share identical columns — the fair-comparison requirement needs
that.

## 7. Example
```cpp
write_measurements_csv("results/vecadd_e2e_calibration.csv",
                       "vecadd", "e2e", calib_rows);
```

## 8. Common mistakes
- Comparing CSVs whose `mode` column differs (e2e vs resident).
- Rows with `valid=0`: exclude them before fitting or plotting.

## 9. Dependencies
Public header + `<fstream>`. Used by `src/main.cpp` and the sample app.
