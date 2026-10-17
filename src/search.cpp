// Adaptive empirical search: no formula, just measurements.
//
// 1. Scan geometrically from start_n (each step x growth_factor) and
//    record every size until the budget or max_n runs out.
// 2. Collect every bracket [a, b] where the faster side changed.
// 3. If a bracket exists, bisect it: measure midpoints, keeping the
//    half that still brackets a sign change, for a few rounds.
// 4. Verify the final candidate with one fresh measurement.
//
// Honesty rules: multiple brackets are all reported; if the GPU never
// wins, found=false; the note always states the stopping reason.

#include "cudagauge/cudagauge.h"

namespace cudagauge {

static bool gpu_won(const Measurement& m) {
    return m.valid && m.gpu_ms < m.cpu_ms;
}

SearchResult adaptive_search(CustomWorkload& work,
                             TimingMode mode,
                             long long start_n,
                             long long max_n,
                             double growth_factor,
                             int repeats,
                             int budget) {
    SearchResult result;
    if (growth_factor <= 1.0) {
        growth_factor = 2.0;
    }
    if (budget < 1) {
        result.note = "empty budget, nothing measured";
        return result;
    }
    // 1. Geometric scan.
    long long n = start_n;
    int steady_gpu_wins = 0;
    while (n <= max_n && (int)result.history.size() < budget) {
        Measurement m = run_custom_size(work, n, mode, repeats);
        result.history.push_back(m);
        if (gpu_won(m)) {
            steady_gpu_wins++;
        } else {
            steady_gpu_wins = 0;
        }
        // Stop early only after two consecutive GPU wins: one win
        // could be noise, two in a row is a usable signal.
        if (steady_gpu_wins >= 2) {
            break;
        }
        long long next = (long long)(n * growth_factor);
        if (next <= n) {
            next = n + 1;
        }
        n = next;
    }
    result.num_measurements = (int)result.history.size();
    // 2. Collect every sign-change bracket in scan order.
    for (size_t i = 1; i < result.history.size(); i++) {
        const Measurement& a = result.history[i - 1];
        const Measurement& b = result.history[i];
        if (!a.valid || !b.valid) {
            continue;
        }
        bool a_gpu = a.gpu_ms < a.cpu_ms;
        bool b_gpu = b.gpu_ms < b.cpu_ms;
        if (a_gpu != b_gpu) {
            SearchInterval bracket;
            bracket.lower_n = a.size_n;
            bracket.upper_n = b.size_n;
            result.transitions.push_back(bracket);
        }
    }
    if (result.transitions.empty()) {
        result.note = "no sign change in scanned range";
        return result;
    }
    // 3. Refine the first bracket by bisection.
    long long low = result.transitions[0].lower_n;
    long long high = result.transitions[0].upper_n;
    for (int round = 0; round < 3; round++) {
        if ((int)result.history.size() + 2 > budget) {
            break;
        }
        long long mid = low + (high - low) / 2;
        if (mid <= low || mid >= high) {
            break;
        }
        Measurement m = run_custom_size(work, mid, mode, repeats);
        result.history.push_back(m);
        result.num_measurements++;
        if (!m.valid) {
            break;
        }
        Measurement low_side = run_custom_size(work, low, mode, 1);
        result.history.push_back(low_side);
        result.num_measurements++;
        bool low_gpu = low_side.gpu_ms < low_side.cpu_ms;
        bool mid_gpu = m.gpu_ms < m.cpu_ms;
        if (low_gpu != mid_gpu) {
            high = mid;
        } else {
            low = mid;
        }
    }
    // 4. Verify with a fresh measurement at the candidate edge.
    result.lower_n = low;
    result.upper_n = high;
    if ((int)result.history.size() >= budget) {
        result.note = "budget exhausted before verification, candidate unverified";
        return result;
    }
    Measurement verify = run_custom_size(work, high, mode, repeats);
    result.history.push_back(verify);
    result.num_measurements++;
    if (verify.valid && verify.gpu_ms < verify.cpu_ms) {
        result.found = true;
        result.crossover_n = high;
        result.note = "verified: GPU faster at upper edge on fresh repeats";
    } else {
        result.note = "candidate did not verify, treating as uncertain";
    }
    return result;
}

}  // namespace cudagauge
