// CudaGauge command-line tool: runs calibration, fits the formula,
// validates on fresh sizes, runs the empirical search, writes CSVs,
// and prints an honest comparison of both methods.
//
// Usage:
//   cudagauge_cli [--workload vecadd|reduce|matmul|all]
//                 [--mode e2e|resident|both]
//                 [--repeats K] [--out DIR]
//
// Calibration sizes fit the model; validation sizes are different
// sizes measured afterwards, so the formula is never tested on the
// data that fitted it.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cudagauge/cudagauge.h"
#include "cudagauge/workloads.h"

using namespace cudagauge;

struct Plan {
    std::string name;
    CustomWorkload (*make)();
    std::vector<long long> calib_sizes;
    std::vector<long long> valid_sizes;
    long long search_start;
    long long search_max;
    double growth;
    int repeats;
};

// First n where the GPU wins twice in a row in sorted measurements.
// Returns -1 when the GPU never wins steadily.
static long long observed_crossover(std::vector<Measurement> rows) {
    // Simple insertion sort by size, then first steady GPU double-win.
    for (size_t i = 1; i < rows.size(); i++) {
        Measurement key = rows[i];
        size_t j = i;
        while (j > 0 && rows[j - 1].size_n > key.size_n) {
            rows[j] = rows[j - 1];
            j--;
        }
        rows[j] = key;
    }
    for (size_t i = 0; i + 1 < rows.size(); i++) {
        bool first = rows[i].valid && rows[i].gpu_ms < rows[i].cpu_ms;
        bool second = rows[i + 1].valid && rows[i + 1].gpu_ms < rows[i + 1].cpu_ms;
        if (first && second) {
            return rows[i].size_n;
        }
    }
    return -1;
}

// Fraction of validation sizes where "use GPU iff n >= threshold"
// picks the actually faster side.
static double threshold_accuracy(const std::vector<Measurement>& rows, long long threshold) {
    int good = 0;
    int total = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        if (!rows[i].valid) {
            continue;
        }
        bool gpu_faster = rows[i].gpu_ms < rows[i].cpu_ms;
        bool pick_gpu = rows[i].size_n >= threshold;
        if (gpu_faster == pick_gpu) {
            good++;
        }
        total++;
    }
    if (total == 0) {
        return 0.0;
    }
    return (double)good / (double)total;
}

static std::vector<long long> sizes_of(const std::vector<Measurement>& rows) {
    std::vector<long long> out;
    for (size_t i = 0; i < rows.size(); i++) {
        out.push_back(rows[i].size_n);
    }
    return out;
}

static std::vector<double> cpu_of(const std::vector<Measurement>& rows) {
    std::vector<double> out;
    for (size_t i = 0; i < rows.size(); i++) {
        out.push_back(rows[i].cpu_ms);
    }
    return out;
}

static std::vector<double> gpu_of(const std::vector<Measurement>& rows) {
    std::vector<double> out;
    for (size_t i = 0; i < rows.size(); i++) {
        out.push_back(rows[i].gpu_ms);
    }
    return out;
}

static void run_plan(Plan& plan, TimingMode mode, const std::string& out_dir) {
    std::string mode_name = mode_to_string(mode);
    std::printf("=== %s [%s] ===\n", plan.name.c_str(), mode_name.c_str());

    CustomWorkload work = plan.make();
    auto cal_start = std::chrono::steady_clock::now();
    std::vector<Measurement> calib = calibrate_custom(work, plan.calib_sizes, mode, plan.repeats);
    double cal_seconds =
        (double)(std::chrono::steady_clock::now() - cal_start).count() / 1e9;

    LinearFit cpu_fit = fit_line(sizes_of(calib), cpu_of(calib));
    LinearFit gpu_fit = fit_line(sizes_of(calib), gpu_of(calib));
    Prediction pred = predict_crossover(cpu_fit, gpu_fit, plan.calib_sizes.front(),
                                        plan.search_max);

    std::vector<Measurement> valid = calibrate_custom(work, plan.valid_sizes, mode, plan.repeats);

    std::vector<Measurement> all = calib;
    for (size_t i = 0; i < valid.size(); i++) {
        all.push_back(valid[i]);
    }
    long long observed = observed_crossover(all);

    SearchResult found =
        adaptive_search(work, mode, plan.search_start, plan.search_max, plan.growth,
                        plan.repeats, 24);

    std::string base = out_dir + "/" + plan.name + "_" + mode_name;
    write_measurements_csv(base + "_calibration.csv", plan.name, mode_name, calib);
    write_measurements_csv(base + "_validation.csv", plan.name, mode_name, valid);
    write_measurements_csv(base + "_search.csv", plan.name, mode_name, found.history);

    std::printf("  cpu fit: t = %.4f + %.6f * n (%s)\n", cpu_fit.intercept, cpu_fit.slope,
                cpu_fit.note.c_str());
    std::printf("  gpu fit: t = %.4f + %.6f * n (%s)\n", gpu_fit.intercept, gpu_fit.slope,
                gpu_fit.note.c_str());
    if (pred.has_crossover) {
        std::printf("  formula predicts n* = %lld%s (%s)\n", pred.crossover_n,
                    pred.extrapolated ? " [EXTRAPOLATED]" : "", pred.note.c_str());
    } else {
        std::printf("  formula predicts no crossover (%s)\n", pred.note.c_str());
    }
    if (observed >= 0) {
        std::printf("  observed steady crossover near n = %lld\n", observed);
    } else {
        std::printf("  observed: GPU never steadily faster in range\n");
    }
    if (pred.has_crossover && observed >= 0) {
        double abs_err = (double)(pred.crossover_n > observed ? pred.crossover_n - observed
                                                              : observed - pred.crossover_n);
        std::printf("  abs error = %.0f, rel error = %.3f\n", abs_err, abs_err / (double)observed);
        std::printf("  formula-threshold accuracy on validation: %.2f\n",
                    threshold_accuracy(valid, pred.crossover_n));
    }
    if (found.found) {
        std::printf("  empirical search: crossover %lld in [%lld, %lld], %d measurements (%s)\n",
                    found.crossover_n, found.lower_n, found.upper_n, found.num_measurements,
                    found.note.c_str());
        std::printf("  search-threshold accuracy on validation: %.2f\n",
                    threshold_accuracy(valid, found.crossover_n));
    } else {
        std::printf("  empirical search: none (%s)\n", found.note.c_str());
    }
    long long baseline = plan.calib_sizes[plan.calib_sizes.size() / 2];
    std::printf("  fixed-baseline n=%lld accuracy on validation: %.2f\n", baseline,
                threshold_accuracy(valid, baseline));
    std::printf("  calibration time: %.1f s, invalid rows: ", cal_seconds);
    int bad = 0;
    for (size_t i = 0; i < all.size(); i++) {
        if (!all[i].valid) {
            bad++;
        }
    }
    std::printf("%d\n", bad);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    std::string want_workload = "all";
    std::string want_mode = "both";
    std::string out_dir = "results";
    int repeats_override = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--workload" && i + 1 < argc) {
            want_workload = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            want_mode = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (arg == "--repeats" && i + 1 < argc) {
            repeats_override = std::atoi(argv[++i]);
        } else {
            std::printf("unknown arg: %s\n", arg.c_str());
            return 1;
        }
    }

    std::vector<Plan> plans;
    Plan vecadd;
    vecadd.name = "vecadd";
    vecadd.make = make_vecadd_workload;
    vecadd.calib_sizes.push_back(4096);
    vecadd.calib_sizes.push_back(16384);
    vecadd.calib_sizes.push_back(65536);
    vecadd.calib_sizes.push_back(262144);
    vecadd.calib_sizes.push_back(1048576);
    vecadd.valid_sizes.push_back(8192);
    vecadd.valid_sizes.push_back(32768);
    vecadd.valid_sizes.push_back(131072);
    vecadd.valid_sizes.push_back(524288);
    vecadd.valid_sizes.push_back(2097152);
    vecadd.search_start = 1024;
    vecadd.search_max = 16777216;
    vecadd.growth = 4.0;
    vecadd.repeats = 5;
    plans.push_back(vecadd);

    Plan reduce = vecadd;
    reduce.name = "reduce";
    reduce.make = make_reduce_workload;
    plans.push_back(reduce);

    Plan matmul;
    matmul.name = "matmul";
    matmul.make = make_matmul_workload;
    matmul.calib_sizes.push_back(32);
    matmul.calib_sizes.push_back(64);
    matmul.calib_sizes.push_back(128);
    matmul.calib_sizes.push_back(256);
    matmul.calib_sizes.push_back(512);
    matmul.valid_sizes.push_back(48);
    matmul.valid_sizes.push_back(96);
    matmul.valid_sizes.push_back(192);
    matmul.valid_sizes.push_back(384);
    matmul.valid_sizes.push_back(768);
    matmul.search_start = 16;
    matmul.search_max = 1024;
    matmul.growth = 2.0;
    matmul.repeats = 3;
    plans.push_back(matmul);

    if (repeats_override > 0) {
        for (size_t i = 0; i < plans.size(); i++) {
            plans[i].repeats = repeats_override;
        }
    }

    std::vector<TimingMode> modes;
    if (want_mode == "e2e" || want_mode == "both") {
        modes.push_back(TimingMode::EndToEnd);
    }
    if (want_mode == "resident" || want_mode == "both") {
        modes.push_back(TimingMode::Resident);
    }
    if (modes.empty()) {
        std::printf("unknown mode: %s\n", want_mode.c_str());
        return 1;
    }

    for (size_t i = 0; i < plans.size(); i++) {
        if (want_workload != "all" && want_workload != plans[i].name) {
            continue;
        }
        for (size_t m = 0; m < modes.size(); m++) {
            run_plan(plans[i], modes[m], out_dir);
        }
    }
    return 0;
}
