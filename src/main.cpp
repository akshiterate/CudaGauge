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

#include <direct.h>  // _mkdir

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

// First n (in elements / dimension, the same N the measurements use)
// where the GPU wins twice in a row in sorted measurements.
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

    // The formula may only see valid measurements: an invalid row (a
    // correctness or allocation failure) carries times that must not
    // shape the model.
    std::vector<Measurement> calib_valid = filter_valid(calib);
    LinearFit cpu_fit = fit_line(sizes_of(calib_valid), cpu_of(calib_valid));
    LinearFit gpu_fit = fit_line(sizes_of(calib_valid), gpu_of(calib_valid));
    Prediction pred = predict_crossover(cpu_fit, gpu_fit, plan.calib_sizes.front(),
                                        plan.search_max);

    std::vector<Measurement> valid = calibrate_custom(work, plan.valid_sizes, mode, plan.repeats);

    std::vector<Measurement> all = calib;
    for (size_t i = 0; i < valid.size(); i++) {
        all.push_back(valid[i]);
    }
    long long observed = observed_crossover(all);

    SearchResult found_out =
        adaptive_search(work, mode, plan.search_start, plan.search_max, plan.growth,
                        plan.repeats, 24);

    std::string base = out_dir + "/" + plan.name + "_" + mode_name;
    write_measurements_csv(base + "_calibration.csv", plan.name, mode_name, calib);
    write_measurements_csv(base + "_validation.csv", plan.name, mode_name, valid);
    write_measurements_csv(base + "_search.csv", plan.name, mode_name, found_out.history);

    std::printf("  valid calibration rows: %d of %d\n", (int)calib_valid.size(),
                (int)calib.size());
    if (!cpu_fit.ok || !gpu_fit.ok) {
        std::printf("  formula rejected: insufficient valid calibration rows (%s / %s)\n",
                    cpu_fit.note.c_str(), gpu_fit.note.c_str());
    }
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
        std::printf("  abs error = %.0f, rel error = %.3f (vs the observed point; the true\n",
                    abs_err, abs_err / (double)observed);
        std::printf("  crossover lies between the neighbouring grid sizes, so this is a\n");
        std::printf("  quantized estimate, not an exact distance)\n");
        std::printf("  formula-threshold accuracy on validation: %.2f\n",
                    threshold_accuracy(valid, pred.crossover_n));
    }
    if (found_out.found) {
        std::printf("  empirical search: %s %lld in [%lld, %lld], %d measurements (%s)\n",
                    outcome_to_string(found_out.outcome), found_out.crossover_n,
                    found_out.lower_n, found_out.upper_n, found_out.num_measurements,
                    found_out.note.c_str());
        std::printf("  search-threshold accuracy on validation: %.2f\n",
                    threshold_accuracy(valid, found_out.crossover_n));
    } else {
        std::printf("  empirical search: none (%s / %s)\n",
                    outcome_to_string(found_out.outcome), found_out.note.c_str());
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

// Refuse to run if any planned output file already exists, so a new run
// can never silently overwrite recorded measurements. --overwrite opts
// out explicitly.
static std::vector<std::string> planned_files(const std::string& out_dir,
                                              const std::vector<Plan>& plans,
                                              const std::vector<TimingMode>& modes) {
    std::vector<std::string> files;
    for (size_t i = 0; i < plans.size(); i++) {
        for (size_t m = 0; m < modes.size(); m++) {
            std::string base = out_dir + "/" + plans[i].name + "_" +
                               mode_to_string(modes[m]);
            files.push_back(base + "_calibration.csv");
            files.push_back(base + "_validation.csv");
            files.push_back(base + "_search.csv");
        }
    }
    return files;
}

static bool any_exists(const std::vector<std::string>& files) {
    for (size_t i = 0; i < files.size(); i++) {
        FILE* f = std::fopen(files[i].c_str(), "rb");
        if (f) {
            std::fclose(f);
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    std::string want_workload = "all";
    std::string want_mode = "both";
    std::string out_dir = "results";
    bool overwrite = false;
    std::string probe_sizes;  // --probe n1,n2,... : extra one-off measurements
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
        } else if (arg == "--overwrite") {
            overwrite = true;
        } else if (arg == "--probe" && i + 1 < argc) {
            probe_sizes = argv[++i];
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

    // P1: never overwrite recorded measurements by default. Pre-flight
    // check on every planned output file (probe files included), so the
    // run is refused before any benchmarking starts.
    std::vector<std::string> files = planned_files(out_dir, plans, modes);
    if (!probe_sizes.empty()) {
        for (size_t i = 0; i < plans.size(); i++) {
            if (want_workload != "all" && want_workload != plans[i].name) {
                continue;
            }
            for (size_t m = 0; m < modes.size(); m++) {
                files.push_back(out_dir + "/" + plans[i].name + "_" +
                                mode_to_string(modes[m]) + "_probe.csv");
            }
        }
    }
    if (!overwrite && any_exists(files)) {
        std::printf("refusing to overwrite existing results in %s\n", out_dir.c_str());
        std::printf("pass --overwrite, or use a fresh --out directory (e.g. results_2026-10-18)\n");
        return 1;
    }
    if (_mkdir(out_dir.c_str()) != 0) {
        // Exists (or cannot create): either fine or the writer will fail
        // loudly on the first CSV.
    }

    for (size_t i = 0; i < plans.size(); i++) {
        if (want_workload != "all" && want_workload != plans[i].name) {
            continue;
        }
        for (size_t m = 0; m < modes.size(); m++) {
            run_plan(plans[i], modes[m], out_dir);
        }
    }

    // --probe n1,n2,... : measure given sizes without fitting or
    // searching. Used to keep extra measurements near a suspected
    // boundary (write into a fresh --out directory).
    if (!probe_sizes.empty()) {
        std::vector<long long> sizes;
        std::string item;
        for (size_t i = 0; i <= probe_sizes.size(); i++) {
            if (i == probe_sizes.size() || probe_sizes[i] == ',') {
                if (!item.empty()) {
                    sizes.push_back(std::atoll(item.c_str()));
                    item.clear();
                }
            } else {
                item += probe_sizes[i];
            }
        }
        for (size_t m = 0; m < modes.size(); m++) {
            for (size_t i = 0; i < plans.size(); i++) {
                if (want_workload != "all" && want_workload != plans[i].name) {
                    continue;
                }
                CustomWorkload work = plans[i].make();
                std::vector<Measurement> rows =
                    calibrate_custom(work, sizes, modes[m], plans[i].repeats);
                for (size_t r = 0; r < rows.size(); r++) {
                    std::printf("[%s] probe n=%lld cpu=%.4f ms gpu=%.4f ms valid=%d\n",
                                mode_to_string(modes[m]).c_str(), rows[r].size_n,
                                rows[r].cpu_ms, rows[r].gpu_ms, (int)rows[r].valid);
                }
                std::string mode_name = mode_to_string(modes[m]);
                if (!write_measurements_csv(
                        out_dir + "/" + plans[i].name + "_" + mode_name + "_probe.csv",
                        plans[i].name, mode_name, rows)) {
                    std::printf("could not write probe CSV in %s\n", out_dir.c_str());
                    return 1;
                }
            }
        }
    }
    return 0;
}
