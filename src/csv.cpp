// One-row CSV writer for benchmark measurements.

#include "cudagauge/cudagauge.h"

#include <fstream>

namespace cudagauge {

bool write_measurements_csv(const std::string& file_path,
                            const std::string& workload_name,
                            const std::string& mode_name,
                            const std::vector<Measurement>& rows) {
    std::ofstream out(file_path);
    if (!out) {
        return false;
    }
    out << "workload,mode,n,cpu_ms,gpu_ms,repeats,valid\n";
    for (size_t i = 0; i < rows.size(); i++) {
        const Measurement& m = rows[i];
        out << workload_name << ",";
        out << mode_name << ",";
        out << m.size_n << ",";
        out << m.cpu_ms << ",";
        out << m.gpu_ms << ",";
        out << m.repeats << ",";
        out << (m.valid ? 1 : 0) << "\n";
    }
    return (bool)out;
}

}  // namespace cudagauge
