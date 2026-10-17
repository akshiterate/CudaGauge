// Unit tests for the linear formula: exact data must recover exact
// parameters, degenerate inputs must fail honestly, and invalid rows
// must never be able to distort a prediction.

#include <cmath>
#include <cstdio>
#include <vector>

#include "cudagauge/cudagauge.h"

using namespace cudagauge;

static int failures = 0;

static void expect_near(double got, double want, double tol, const char* label) {
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %s: got %f want %f\n", label, got, want);
        failures++;
    } else {
        std::printf("PASS %s\n", label);
    }
}

static LinearFit cpu_fit_of(const std::vector<Measurement>& rows) {
    std::vector<long long> sizes;
    std::vector<double> times;
    for (size_t i = 0; i < rows.size(); i++) {
        sizes.push_back(rows[i].size_n);
        times.push_back(rows[i].cpu_ms);
    }
    return fit_line(sizes, times);
}

static LinearFit gpu_fit_of(const std::vector<Measurement>& rows) {
    std::vector<long long> sizes;
    std::vector<double> times;
    for (size_t i = 0; i < rows.size(); i++) {
        sizes.push_back(rows[i].size_n);
        times.push_back(rows[i].gpu_ms);
    }
    return fit_line(sizes, times);
}

int main() {
    // y = 1 + 2x exactly.
    std::vector<long long> xs;
    std::vector<double> ys;
    xs.push_back(10);
    xs.push_back(20);
    xs.push_back(30);
    xs.push_back(40);
    ys.push_back(21.0);
    ys.push_back(41.0);
    ys.push_back(61.0);
    ys.push_back(81.0);
    LinearFit fit = fit_line(xs, ys);
    expect_near(fit.slope, 2.0, 1e-9, "slope of exact line");
    expect_near(fit.intercept, 1.0, 1e-9, "intercept of exact line");

    // cpu = 1 + 2x, gpu = 5 + 1x -> cross at x = 4.
    LinearFit cpu;
    cpu.ok = true;
    cpu.intercept = 1.0;
    cpu.slope = 2.0;
    LinearFit gpu;
    gpu.ok = true;
    gpu.intercept = 5.0;
    gpu.slope = 1.0;
    Prediction p = predict_crossover(cpu, gpu, 1, 100);
    if (!p.has_crossover || p.crossover_n != 4 || p.extrapolated) {
        std::printf("FAIL crossover at 4\n");
        failures++;
    } else {
        std::printf("PASS crossover at 4\n");
    }

    // Parallel lines: no crossover.
    LinearFit gpu2 = gpu;
    gpu2.slope = 2.0;
    Prediction p2 = predict_crossover(cpu, gpu2, 1, 100);
    if (p2.has_crossover) {
        std::printf("FAIL parallel lines should have no crossover\n");
        failures++;
    } else {
        std::printf("PASS parallel lines\n");
    }

    // Single point: fit impossible.
    std::vector<long long> one;
    std::vector<double> one_y;
    one.push_back(5);
    one_y.push_back(3.0);
    if (fit_line(one, one_y).ok) {
        std::printf("FAIL single point should not fit\n");
        failures++;
    } else {
        std::printf("PASS single point rejected\n");
    }

    // Extrapolation flag: cross at 4 but range starts at 10.
    Prediction p3 = predict_crossover(cpu, gpu, 10, 100);
    if (!p3.has_crossover || !p3.extrapolated) {
        std::printf("FAIL extrapolation flag\n");
        failures++;
    } else {
        std::printf("PASS extrapolation flag\n");
    }

    // --- P2: invalid rows can never influence a prediction ---
    // Clean data: (x, cpu=y=1+2x, gpu=5+1x) -> crossover at x=4.
    std::vector<Measurement> clean;
    for (size_t i = 0; i < xs.size(); i++) {
        Measurement m;
        m.size_n = xs[i];
        m.cpu_ms = ys[i];                    // 21, 41, 61, 81 (y = 1+2x)
        m.gpu_ms = 5.0 + (double)xs[i];      // gpu = 5 + 1x
        m.valid = true;
        clean.push_back(m);
    }
    Prediction clean_pred = predict_crossover(cpu_fit_of(clean), gpu_fit_of(clean), 1, 100);

    // Contaminated data: an extra invalid row with absurd times must be
    // dropped by filter_valid and leave the prediction untouched.
    std::vector<Measurement> contaminated = clean;
    Measurement bad;
    bad.size_n = 9999999;
    bad.cpu_ms = 1.0e9;
    bad.gpu_ms = 2.0e9;
    bad.valid = false;  // correctness failed -> unusable by definition
    contaminated.push_back(bad);
    std::vector<Measurement> kept = filter_valid(contaminated);
    if (kept.size() != clean.size()) {
        std::printf("FAIL filter_valid size: got %d want %d\n", (int)kept.size(),
                    (int)clean.size());
        failures++;
    } else {
        std::printf("PASS filter_valid drops the invalid row\n");
    }
    Prediction contaminated_pred =
        predict_crossover(cpu_fit_of(kept), gpu_fit_of(kept), 1, 100);
    if (contaminated_pred.has_crossover != clean_pred.has_crossover ||
        contaminated_pred.crossover_n != clean_pred.crossover_n ||
        contaminated_pred.extrapolated != clean_pred.extrapolated) {
        std::printf("FAIL prediction changed after filtering a bad row\n");
        failures++;
    } else {
        std::printf("PASS invalid row cannot influence a prediction\n");
    }

    // Insufficient valid data: with only one valid row the fit must be
    // refused and the prediction must report the refusal honestly.
    std::vector<Measurement> scarce;
    Measurement only;
    only.size_n = 64;
    only.cpu_ms = 130.0;
    only.gpu_ms = 70.0;
    only.valid = true;
    scarce.push_back(only);
    scarce.push_back(bad);
    LinearFit scarce_fit = cpu_fit_of(filter_valid(scarce));
    if (scarce_fit.ok) {
        std::printf("FAIL fit accepted a single valid point\n");
        failures++;
    } else {
        std::printf("PASS insufficient valid calibration data rejected\n");
    }
    Prediction scarce_pred = predict_crossover(cpu_fit_of(filter_valid(scarce)),
                                               gpu_fit_of(filter_valid(scarce)), 1, 100);
    if (scarce_pred.has_crossover) {
        std::printf("FAIL prediction produced from a refused fit\n");
        failures++;
    } else {
        std::printf("PASS no prediction from insufficient data\n");
    }

    if (failures > 0) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all model tests passed\n");
    return 0;
}
