# Results directory — provenance of every run

Rule since 2026-10-18: each benchmark run must write into a unique,
clearly named subdirectory. `cudagauge_cli` refuses to overwrite an
existing output file unless `--overwrite` is given, so recorded
measurements are never silently replaced.

| Directory | Provenance |
|---|---|
| `2026-10-15_historical/console_log.txt` | Console output of the original experiment run (RTX 2050, driver 555.97, CUDA 12.5, Release build). This is the only surviving artifact of that run and the source of the table in `explanations/EXPERIMENT_LOG.md` (formula 85304 / 32852 / 92 / 91, observed 262144 / 32768 / 48 / 48). **Its CSV files were overwritten during the 2026-10-17 rebuild and are unrecoverable.** |
| `2026-10-17_regen_run/*.csv` | Full raw CSV output of the 2026-10-17 rename-verification regeneration: 3 workloads x 2 modes x (calibration / validation / search) = 18 built-in CSVs, 6 third-party vectorAdd CSVs, and `sample_app.csv`. Verified: every row has `valid=1`. Headline values of that run (for cross-checking against the historical table): vecadd-resident formula 126770 vs observed 262144 (rel 0.516); reduce-resident formula 78116 vs observed 32768 (rel 1.384); matmul-e2e formula 98 vs observed 64; matmul-resident formula 90 vs observed 48. Search outcomes that differ from the historical run are recorded verbatim in the conversation log of that date and, from 2026-10-18 on, in each run's own directory. |
| `2026-10-18_reduce_e2e_recheck/` | Re-evaluation of the unstable `reduce` end-to-end result (historical log claimed a verified crossover at the 16.7M edge; the 2026-10-17 run showed the GPU losing there). Contains the full plan CSVs plus `_probe.csv` with extra measurements near the upper boundary. See `explanations/EXPERIMENT_LOG.md` for the verdict — written down as observed uncertainty, not as an established crossover. |

Notes:
- Raw CSVs are not tracked by default (`.gitignore`: `results/*.csv`).
  Once a run is validated, force-add its directory to preserve it:
  `git add -f results/<run-dir>`.
- Root-level writes (the old default `results/*.csv`) are what made the
  2026-10-15 CSV loss possible; the overwrite guard now makes that
  pattern impossible without `--overwrite`.
- `sample_app.**` always writes `results/sample_app.csv` (Level 2
  example keeps the simple path; it is regenerated per rebuild).
