# `src/search.cpp` — adaptive empirical search (no formula)

## 1. Purpose
Implements `adaptive_search`: finds where the GPU becomes faster using
only fresh measurements — geometric scan, bracket collection,
bisection refinement, verification. Never calls the linear model.

## 2. Concepts
- **Geometric sampling**: multiply size by a factor (e.g. 4×) each step.
  Covers 1024 → 16M in 8 steps instead of thousands of linear steps.
- **Bracket**: adjacent measured sizes where different sides win.
  The crossover lies somewhere inside.
- **Bisection**: repeatedly measure the midpoint and keep the half that
  still brackets a sign change; 3 rounds shrink the interval 8×.
- **Verification**: one extra fresh measurement at the candidate; noise
  near the boundary can flip a single reading, so the search refuses to
  confirm shaky candidates.

## 3–5. Walkthrough
- `gpu_won(m)` helper: `valid && gpu_ms < cpu_ms`. Invalid rows never
  count as wins.
- Scan loop: measure `start_n`, `start_n*growth`, … until `max_n`,
  budget spent, or **two consecutive GPU wins** (one win could be
  noise; two in a row is the stop signal). `next <= n` guard prevents
  infinite loops when `growth` rounds down (tiny sizes).
- Bracket collection: every adjacent pair with different winners is
  recorded in `transitions` — all of them, so non-monotonic curves show
  multiple brackets instead of one forced answer.
- Refinement: 3 bisection rounds on the *first* bracket, each measuring
  the midpoint (full repeats) and re-measuring the low edge (1 repeat,
  cheap anchor). Stops early if the budget cannot cover a full round.
- Verification: fresh full-repeat measurement at the upper edge.
  Confirmed → `found=true`; else honest `note`, `found=false`. If the
  budget ran out first, that is reported instead of guessing.
- `num_measurements` counts every size measured (scan + refine +
  verify), so cost comparisons with the formula are fair.

## 6. Why this design
The two-win stop, the kept history, and the verify-or-admit-uncertain
rule all serve one goal: the search must be trustworthy when the
formula is wrong, including when the answer is "no crossover".

## 7. Example
```cpp
SearchResult s = adaptive_search(w, TimingMode::Resident, 1024,
                                 16777216, 4.0, 5, 24);
// s.found, s.crossover_n, s.lower_n/upper_n bracket, s.history rows.
```

## 8. Common mistakes
- Tiny budget (scan alone needs ~6–8) — check `num_measurements`
  against the budget to see if refinement even ran.
- Microsecond-scale timings: at ~0.05 ms, noise exceeds the gap and
  bisection coin-flips (seen in reduce-resident). Widen repeats or
  accept "uncertain".
- Assuming one bracket: always inspect `transitions`.

## 9. Dependencies
Public header + `run_custom_size` from `src/benchmark.cu`. Tested by
`tests/test_search.cpp`, used by CLI and sample app.
