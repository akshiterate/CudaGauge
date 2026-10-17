// Small helpers with no CUDA dependency: median and mode names.

#include "cudagauge/cudagauge.h"

#include <algorithm>

namespace cudagauge {

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

std::string mode_to_string(TimingMode mode) {
    if (mode == TimingMode::EndToEnd) {
        return "e2e";
    }
    return "resident";
}

const char* outcome_to_string(SearchOutcome outcome) {
    switch (outcome) {
        case SearchOutcome::VerifiedInterior:
            return "verified-interior";
        case SearchOutcome::BoundaryCandidate:
            return "boundary-candidate";
        case SearchOutcome::Unverified:
            return "unverified";
        case SearchOutcome::BudgetExhausted:
            return "budget-exhausted";
        default:
            return "none";
    }
}

std::vector<Measurement> filter_valid(const std::vector<Measurement>& rows) {
    std::vector<Measurement> out;
    out.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].valid) {
            out.push_back(rows[i]);
        }
    }
    return out;
}

}  // namespace cudagauge
