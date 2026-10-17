// Formula-based predictor: fit time = a + b * n by ordinary least
// squares, then solve cpu(n) == gpu(n) for the crossover size.
//
// Math used here:
//   slope     = (m * Sxy - Sx * Sy) / (m * Sxx - Sx * Sx)
//   intercept = (Sy - slope * Sx) / m
// where Sx, Sy, Sxy, Sxx are sums over the m calibration points.
// Assumption: timing grows linearly with n. That is only a hypothesis;
// the empirical search in search.cpp never trusts it.

#include "cudagauge/cudagauge.h"

#include <cmath>

namespace cudagauge {

LinearFit fit_line(const std::vector<long long>& sizes,
                   const std::vector<double>& times) {
    LinearFit result;
    if (sizes.size() < 2 || sizes.size() != times.size()) {
        result.note = "need at least 2 points";
        return result;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    for (size_t i = 0; i < sizes.size(); i++) {
        double x = (double)sizes[i];
        double y = times[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }
    double count = (double)sizes.size();
    double denom = count * sum_xx - sum_x * sum_x;
    if (denom == 0.0) {
        result.note = "all sizes identical, slope undefined";
        return result;
    }
    result.slope = (count * sum_xy - sum_x * sum_y) / denom;
    result.intercept = (sum_y - result.slope * sum_x) / count;
    result.ok = true;
    result.note = "least-squares fit";
    return result;
}

Prediction predict_crossover(const LinearFit& cpu_fit,
                             const LinearFit& gpu_fit,
                             long long min_n,
                             long long max_n) {
    Prediction result;
    if (!cpu_fit.ok || !gpu_fit.ok) {
        result.note = "unfitted model, no prediction";
        return result;
    }
    // Solve a + b*n == c + d*n  ->  n = (c - a) / (b - d).
    double denom = cpu_fit.slope - gpu_fit.slope;
    if (std::fabs(denom) < 1e-15) {
        result.note = "lines are parallel, no crossover";
        return result;
    }
    double cross = (gpu_fit.intercept - cpu_fit.intercept) / denom;
    if (cross < 0.0) {
        result.note = "crossover is negative, GPU always faster or never";
        return result;
    }
    result.has_crossover = true;
    result.crossover_n = (long long)(cross + 0.5);
    if (result.crossover_n < min_n || result.crossover_n > max_n) {
        result.extrapolated = true;
        result.note = "prediction outside measured range (extrapolation)";
    } else {
        result.note = "prediction inside measured range";
    }
    return result;
}

}  // namespace cudagauge
