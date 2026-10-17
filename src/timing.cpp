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

}  // namespace cudagauge
