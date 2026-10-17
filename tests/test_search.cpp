// Unit test for the adaptive search using a synthetic workload with
// no GPU code: the CPU side busy-loops proportional to n, the GPU side
// busy-loops with a fixed cost plus a smaller slope, so a real
// crossover exists near n = 10000 and the search must find a bracket.
//
// Additional scenario tests use deterministic piecewise/stateful fake
// workloads (decisions depend on n thresholds or run parity, never on
// timing luck) to cover: no bracket at all, a verified candidate at the
// boundary of the search window, multiple observed transitions, noisy
// verification (fresh batches disagree), and an exhausted budget.

#include <cstdio>

#include "cudagauge/cudagauge.h"

using namespace cudagauge;

static int failures = 0;

static long long current_n = 0;

// Incremented once per run_custom_size() call (from setup), so fake
// GPU cost can alternate deterministically between measurements.
static int run_id = 0;

// Burn roughly ops simple operations so timing scales with op count.
static void burn(long long ops) {
    volatile double acc = 0.0;
    for (long long i = 0; i < ops; i++) {
        acc += (double)(i % 7) * 0.5;
    }
    (void)acc;
}

// The shared fake workload builder uses capturing lambdas over the
// file-static current_n (globals need no capture).
static CustomWorkload make_linear(long long cpu_per_n, long long gpu_fixed,
                                  long long gpu_per_n) {
    CustomWorkload w;
    w.name = "fake";
    w.setup = [](long long n) { current_n = n; };
    w.cpu_work = [cpu_per_n]() { burn(current_n * cpu_per_n); };
    w.gpu_e2e_work = [gpu_fixed, gpu_per_n]() { burn(gpu_fixed + current_n * gpu_per_n); };
    w.gpu_kernel_work = w.gpu_e2e_work;
    w.teardown = []() {};
    w.validate = []() { return true; };
    return w;
}

int main() {
    // --- Scenario 0: the original synthetic crossover -------------------
    // cpu = n*5000, gpu = 50M + n*50: crossover near n = 10000.
    CustomWorkload fake;
    fake.name = "fake";
    fake.setup = [](long long n) { current_n = n; };
    fake.cpu_work = []() { burn(current_n * 5000); };
    fake.gpu_e2e_work = []() { burn(50000000 + current_n * 50); };
    fake.gpu_kernel_work = fake.gpu_e2e_work;
    fake.teardown = []() {};
    fake.validate = []() { return true; };

    SearchResult found = adaptive_search(fake, TimingMode::EndToEnd, 100, 1000000, 4.0, 3, 16);
    std::printf("found=%d crossover=%lld bracket=[%lld,%lld] measures=%d note=%s\n",
                (int)found.found, found.crossover_n, found.lower_n, found.upper_n,
                found.num_measurements, found.note.c_str());
    if (!found.found || found.outcome != SearchOutcome::VerifiedInterior ||
        !found.verified || found.at_boundary) {
        std::printf("FAIL search should find the synthetic crossover (interior, verified)\n");
        failures++;
    } else {
        std::printf("PASS synthetic crossover verified interior\n");
    }
    if (found.lower_n >= found.upper_n || found.upper_n <= 0) {
        std::printf("FAIL bracket is not a real interval\n");
        failures++;
    } else {
        std::printf("PASS bracket is a real interval\n");
    }

    // --- Scenario 1: no bracket anywhere (GPU always loses) -------------
    {
        CustomWorkload w = make_linear(200, 30000000, 200);
        SearchResult r = adaptive_search(w, TimingMode::EndToEnd, 1024, 65536, 4.0, 2, 24);
        std::printf("outcome=%s measures=%d note=%s\n", outcome_to_string(r.outcome),
                    r.num_measurements, r.note.c_str());
        if (r.found || r.outcome != SearchOutcome::None || r.verified ||
            !r.transitions.empty()) {
            std::printf("FAIL no-bracket scenario\n");
            failures++;
        } else {
            std::printf("PASS no-bracket reported as none\n");
        }
    }

    // --- Scenario 2: verified candidate at the window boundary ----------
    // GPU loses at every n < 16384 (fixed 30M ops) and wins decisively
    // at n >= 16384. max_n = 16384, so any verified candidate sits on
    // the boundary and must be labelled a boundary candidate.
    {
        CustomWorkload w;
        w.name = "boundary";
        w.setup = [](long long n) { current_n = n; };
        w.cpu_work = []() { burn(current_n * 1000); };
        w.gpu_e2e_work = []() {
            if (current_n >= 16384) {
                burn(50000);
            } else {
                burn(30000000 + current_n * 10);
            }
        };
        w.gpu_kernel_work = w.gpu_e2e_work;
        w.teardown = []() {};
        w.validate = []() { return true; };
        SearchResult r = adaptive_search(w, TimingMode::EndToEnd, 4096, 16384, 4.0, 2, 24);
        std::printf("outcome=%s crossover=%lld bracket=[%lld,%lld] measures=%d note=%s\n",
                    outcome_to_string(r.outcome), r.crossover_n, r.lower_n, r.upper_n,
                    r.num_measurements, r.note.c_str());
        if (!r.found || r.outcome != SearchOutcome::BoundaryCandidate || !r.at_boundary ||
            r.crossover_n != 16384) {
            std::printf("FAIL boundary scenario\n");
            failures++;
        } else {
            std::printf("PASS verified boundary candidate distinguished from an "
                        "interior crossover\n");
        }
    }

    // --- Scenario 3: multiple observed transitions ----------------------
    // GPU wins at exactly one geometric grid point (n == 16384); the
    // lose branch costs 3x the CPU's work per element, so the GPU loses
    // both below and above the cheap point. The scan yields two sign
    // changes: CPU->GPU into it and GPU->CPU out of it. The first
    // bracket is refined and verified; both transitions must be
    // recorded and the note must flag the unresolved rest.
    {
        CustomWorkload w;
        w.name = "multi";
        w.setup = [](long long n) { current_n = n; };
        w.cpu_work = []() { burn(current_n * 1000); };
        w.gpu_e2e_work = []() {
            if (current_n == 16384) {
                burn(50000);
            } else {
                burn(current_n * 3000);
            }
        };
        w.gpu_kernel_work = w.gpu_e2e_work;
        w.teardown = []() {};
        w.validate = []() { return true; };
        SearchResult r =
            adaptive_search(w, TimingMode::EndToEnd, 1024, 262144, 4.0, 2, 32);
        std::printf("outcome=%s crossover=%lld transitions=%d measures=%d note=%s\n",
                    outcome_to_string(r.outcome), r.crossover_n,
                    (int)r.transitions.size(), r.num_measurements, r.note.c_str());
        if (r.transitions.size() < 2) {
            std::printf("FAIL expected at least two observed transitions\n");
            failures++;
        } else {
            std::printf("PASS multiple transitions recorded\n");
        }
        if (!r.found || r.outcome != SearchOutcome::VerifiedInterior ||
            r.note.find("multiple brackets") == std::string::npos) {
            std::printf("FAIL first-of-multiple-brackets handling\n");
            failures++;
        } else {
            std::printf("PASS first bracket resolved, later ones flagged in the note\n");
        }
    }

    // --- Scenario 4: noisy evidence cannot pass verification ------------
    // GPU cost alternates deterministically between consecutive search
    // measurements (odd run_id: GPU wins, even run_id: GPU loses), so
    // the scan sees flip-flopping transitions and, once a candidate
    // exists, the two fresh verification batches disagree by
    // construction. The result must be honest: unverified.
    run_id = 0;
    {
        CustomWorkload w;
        w.name = "noisy";
        w.setup = [](long long n) {
            current_n = n;
            run_id++;
        };
        w.cpu_work = []() { burn(current_n * 1000 + 5000); };
        w.gpu_e2e_work = []() {
            if (run_id % 2 == 1) {
                burn(50000);
            } else {
                burn(40000000);
            }
        };
        w.gpu_kernel_work = w.gpu_e2e_work;
        w.teardown = []() {};
        w.validate = []() { return true; };
        SearchResult r = adaptive_search(w, TimingMode::EndToEnd, 16, 512, 2.0, 2, 14);
        std::printf("outcome=%s verified=%d transitions=%d measures=%d note=%s\n",
                    outcome_to_string(r.outcome), (int)r.verified,
                    (int)r.transitions.size(), r.num_measurements, r.note.c_str());
        if (r.found || r.verified || r.outcome != SearchOutcome::Unverified) {
            std::printf("FAIL noisy verification should end unverified\n");
            failures++;
        } else {
            std::printf("PASS disagreeing verification batches -> unverified\n");
        }
    }

    // --- Scenario 5: budget exhausted before any verification -----------
    run_id = 0;
    {
        CustomWorkload w;
        w.name = "budget";
        w.setup = [](long long n) {
            current_n = n;
            run_id++;
        };
        w.cpu_work = []() { burn(current_n * 1000 + 5000); };
        w.gpu_e2e_work = []() {
            if (run_id % 2 == 1) {
                burn(50000);
            } else {
                burn(40000000);
            }
        };
        w.gpu_kernel_work = w.gpu_e2e_work;
        w.teardown = []() {};
        w.validate = []() { return true; };
        SearchResult r = adaptive_search(w, TimingMode::EndToEnd, 16, 512, 2.0, 2, 6);
        std::printf("outcome=%s verified=%d measures=%d note=%s\n",
                    outcome_to_string(r.outcome), (int)r.verified, r.num_measurements,
                    r.note.c_str());
        if (r.found || r.verified || r.outcome != SearchOutcome::BudgetExhausted ||
            r.num_measurements != 6) {
            std::printf("FAIL exhausted budget should stop before verification\n");
            failures++;
        } else {
            std::printf("PASS exhausted budget reported, budget respected\n");
        }
    }

    if (failures > 0) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all search tests passed\n");
    return 0;
}
