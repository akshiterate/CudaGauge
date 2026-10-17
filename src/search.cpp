// Adaptive empirical search: no formula, just measurements.
//
// 1. Scan geometrically from start_n (each step x growth_factor) and
//    record every size until the budget or max_n runs out.
// 2. Collect every bracket [a, b] where the faster side changed.
// 3. If a bracket exists, bisect the FIRST one: measure midpoints,
//    keeping the half that still brackets a sign change, for a few
//    rounds. Later brackets stay listed in `transitions` but are NOT
//    refined - the budget is spent on resolving one answer. Consequence
//    documented honestly: a second, real crossover later in the range
//    can be missed once an early double-win stops the scan.
// 4. Verify the final candidate with up to two fresh repeat batches.
//    Every batch must show the GPU winning; a single disagreeing batch
//    marks the candidate unverified. This is resistant to noise without
//    pretending noise can be eliminated.
//
// Honesty rules: multiple brackets are all reported; if the GPU never
// wins, outcome=None; structured `outcome` fields distinguish verified
// interior, boundary candidate, unverified, and exhausted-budget cases;
// the note always states the stopping reason.

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
    // 4. Verify the candidate edge with fresh repeat batches. Every
    // batch is one new Measurement (a fresh median) and counts against
    // the budget. All batches must agree that the GPU wins; if any
    // disagrees, the candidate is reported unverified rather than
    // guessing which batch was wrong.
    result.lower_n = low;
    result.upper_n = high;
    if ((int)result.history.size() >= budget) {
        result.outcome = SearchOutcome::BudgetExhausted;
        result.note = "budget exhausted before verification, candidate unverified";
        return result;
    }
    const int max_batches = 2;
    int passed = 0;
    int attempted = 0;
    while (attempted < max_batches && (int)result.history.size() < budget) {
        Measurement verify = run_custom_size(work, high, mode, repeats);
        result.history.push_back(verify);
        result.num_measurements++;
        attempted++;
        if (verify.valid && verify.gpu_ms < verify.cpu_ms) {
            passed++;
        } else {
            break;  // first disagreement decides: noisy evidence is not proof
        }
    }
    result.verified = (passed == attempted && attempted > 0);
    if (!result.verified) {
        result.note = "fresh verification batches did not all confirm the GPU win";
        if (result.transitions.size() > 1) {
            result.note += " (first of multiple brackets)";
        }
        result.outcome = SearchOutcome::Unverified;
        return result;
    }
    result.found = true;
    result.crossover_n = high;
    if (high >= max_n) {
        result.outcome = SearchOutcome::BoundaryCandidate;
        result.at_boundary = true;
        result.note = "verified GPU win at the edge of the search window; the "
                      "true crossover may lie at or beyond max_n";
    } else {
        result.outcome = SearchOutcome::VerifiedInterior;
        result.note = "verified: GPU won all fresh verification batches at the "
                      "upper edge on fresh repeats";
    }
    if (result.transitions.size() > 1) {
        result.note += " (first of multiple brackets)";
    }
    return result;
}

}  // namespace cudagauge
