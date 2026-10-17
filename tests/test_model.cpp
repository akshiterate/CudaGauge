// Unit tests for the linear formula: exact data must recover exact
// parameters, and degenerate inputs must fail honestly.

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

    if (failures > 0) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all model tests passed\n");
    return 0;
}
