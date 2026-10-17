# Experiment log (actual runs — illustrative numbers appear nowhere)

## Hardware / software
- GPU: NVIDIA GeForce RTX 2050 laptop, 4096 MiB, driver 555.97, CUDA 12.5.
- Arch compiled: sm_86. CPU: (laptop, background activity normal).
- Build: Release (MSVC `/O2` default), NMake generator under vcvars64.
- Protocol: repeats 5 (vecadd/reduce), 3 (matmul); medians reported;
  5 calibration + 5 interleaved validation sizes; search budget 24.
- Command: `build\cudagauge_cli.exe --out results` (full console output
  in `results/console_log.txt`); `build\sample_app.exe` for Level 2.

## Results by workload × mode (RTX 2050, 2026-10-15)

| workload | mode | formula n\* | observed n | abs/rel err | search n\* | search acc | formula acc |
|---|---|---|---|---|---|---|---|
| vecadd | e2e | none (neg.) | none in range | — | none | — | — |
| vecadd | resident | 85304 | 262144 | 176840 / 0.675 | 163840 ✓ | 1.00 | 0.80 |
| reduce | e2e | none (neg.) | none ≤ 2M | — | 16777216 (edge) ✓ | 1.00 | — |
| reduce | resident | 32852 | 32768 | 84 / 0.003 | unverified | — | 0.80 |
| matmul | e2e | 92 | 48 | 44 / 0.917 | 64 ✓ | 0.80 | 0.80 |
| matmul | resident | 91 | 48 | 43 / 0.896 | unverified | — | 0.80 |

Fixed middle-size baseline accuracy on validation: 0.40–0.80 (never best).

## Interpretations
1. **vecadd e2e: GPU never wins.** Fitted GPU slope (≈2e-6 ms/N) exceeds
   CPU slope (≈1e-6): three full memory passes per N cannot beat one
   CPU pass. Formula and search agree — correctly reporting *absence*.
2. **reduce resident: formula nearly exact (0.3% error).** Streaming
   kernel time really is linear in N here; the hypothesis holds.
3. **vecadd resident: formula off by 68%.** GPU time is flat
   launch-overhead-dominated, not linear; extrapolation from small-N
   medians misses. Search (measurement, not model) brackets it.
4. **matmul: formula structurally wrong scale.** Cost ∝ dim³ was fitted
   as ∝ dim; predictions 91–92 vs observed 48. Search still finds 64
   (e2e). Verdict: fit polynomials in the right variable or don't use
   the formula — exactly the "investigate differences" the spec asks for.
5. **Two "unverified" searches are data, not bugs.** reduce-resident and
   matmul-resident refine into ~0.04–0.09 ms noise (see
   `reduce_resident_search.csv` rows 11–13: same n=22528 reads 0.039 and
   0.088 ms). The search refused to confirm — the honest output.
6. **reduce e2e edge crossover at 16M** (max size, verified): beyond 2M
   elements even transfer-inclusive GPU reduction wins; inside the
   calib/valid window (≤2M) nothing wins — consistent, not contradictory.
7. **Sample app (scale_shift, e2e):** GPU never wins to 4M elements
   (cpu 2.9 ms vs gpu 6.8 ms) — memory-bound, same lesson as vecadd e2e.

## Conclusion
Neither method wins universally: the formula is precise when linearity
holds (reduce-resident) and misleading otherwise (matmul, flat
overhead regimes); the empirical search is reliable wherever the signal
beats noise but honestly abstains in noise-dominated regimes. A
threshold estimator should use the formula only after a linearity check,
and fall back to the search otherwise.

## Level 3: NVIDIA cuda-samples vectorAdd (RTX 2050, 2026-10-16)
- Adapter: `examples/third_party/vectorAdd/vectorAdd_cudagauge.cu`;
  upstream kernel/CPU/epsilon reused; 6 CSVs in `results/thirdparty_*`.
- **e2e**: formula none, search none, all rows valid — transfer-inclusive
  GPU never wins to 2M elements. Same verdict as built-in vecadd-e2e.
- **resident**: formula n\* = 113793 (in range); search verified 163840.
  Built-in vecadd-resident gave 85304 / 163840: independent upstream
  code reproduces the library result within one grid step — the
  strongest evidence the adapter boundary is sound.
- Compat notes: `cuda::ceil_div` missing in CUDA 12.5 (replaced with
  identical arithmetic); `cudaMemPrefetchAsync` fails (code 101) on this
  WDDM setup, so e2e transfers use `cudaMemcpy`; CPU uses host mirrors
  to avoid migration artifacts. Full detail in the integration README.

## Reproduce
Rebuild (see `build.md`), run `build\cudagauge_cli.exe --out results`,
compare new CSVs against the committed ones. Expect ±20% jitter on
small-N medians (clocks, thermals, background CPU); crossovers may shift
one grid step. Never compare Debug-build numbers with these.
