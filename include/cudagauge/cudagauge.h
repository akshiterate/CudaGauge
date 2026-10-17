#pragma once

// CudaGauge public library API.
//
// A developer who already has a CPU implementation and a CUDA GPU
// implementation describes both through the small CustomWorkload struct
// below, then asks the library to benchmark, model, and search.
//
// The library never modifies the developer's dispatch logic. It only
// measures and recommends a size threshold.
//
// This header intentionally includes no CUDA headers, so a developer's
// plain C++ file can include it without a CUDA include path.

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cudagauge {

// Which costs count in a GPU measurement.
enum class TimingMode {
    EndToEnd,  // host-to-device copy + kernel + device-to-host copy
    Resident   // kernel execution only, input already on the device
};

// One benchmarked workload size.
struct Measurement {
    long long size_n = 0;   // workload size (elements, or matrix dim)
    double cpu_ms = 0.0;    // median CPU time over repeats
    double gpu_ms = 0.0;    // median GPU time over repeats
    int repeats = 0;        // how many timed repeats produced the medians
    bool valid = false;     // false when correctness check or alloc failed
};

// Adapter between the library and a developer's own code.
//
// setup(n) allocates and fills inputs for size n. It is never timed.
// cpu_work() runs one CPU pass over the current inputs. It is timed.
// gpu_e2e_work() runs one end-to-end GPU pass. It is timed in EndToEnd mode.
// gpu_kernel_work() runs one kernel-only GPU pass. It is timed in Resident.
// teardown() releases everything from the last setup().
// validate() checks the last CPU and GPU outputs agree. Untimed.
struct CustomWorkload {
    std::string name;
    std::function<void(long long n)> setup;
    std::function<void()> cpu_work;
    std::function<void()> gpu_e2e_work;
    std::function<void()> gpu_kernel_work;
    std::function<void()> teardown;
    std::function<bool()> validate;
};

// A fitted line: time = intercept + slope * n.
struct LinearFit {
    double intercept = 0.0;  // fixed cost in milliseconds
    double slope = 0.0;      // per-unit cost in milliseconds
    bool ok = false;         // false when the fit was impossible
    std::string note;        // human-readable explanation
};

// A predicted crossover size where the GPU becomes faster than the CPU.
struct Prediction {
    bool has_crossover = false;
    long long crossover_n = 0;
    bool extrapolated = false;  // true when outside the measured range
    std::string note;
};

// One interval where the faster side changed between two measured sizes.
struct SearchInterval {
    long long lower_n = 0;
    long long upper_n = 0;
};

// Outcome of the adaptive empirical search.
struct SearchResult {
    bool found = false;
    long long crossover_n = 0;  // smallest size with confirmed GPU win
    long long lower_n = 0;      // last size where the CPU still won
    long long upper_n = 0;      // first size where the GPU won
    int num_measurements = 0;   // how many sizes were measured
    std::string note;
    std::vector<Measurement> history;
    std::vector<SearchInterval> transitions;  // every observed sign change
};

// Median of a copy of values. Used instead of a mean so one slow
// outlier repeat cannot drag the reported time upwards.
double median_of(std::vector<double> values);

// Human-readable timing mode name for CSV files and log lines.
std::string mode_to_string(TimingMode mode);

// Abort-safe CUDA error check. The code is a cudaError_t passed as an
// int so this header stays free of CUDA includes. Call sites live in
// .cu files and cast with (int)err. Prints file/line and exits.
void cuda_check(int error_code, const char* what, const char* file, int line);

// Benchmark one workload size: warm up, time repeats, validate.
Measurement run_custom_size(CustomWorkload& work,
                            long long n,
                            TimingMode mode,
                            int repeats);

// Benchmark every size in order. Each size gets a fresh setup/teardown
// so sizes do not leak state into each other.
std::vector<Measurement> calibrate_custom(CustomWorkload& work,
                                           const std::vector<long long>& sizes,
                                           TimingMode mode,
                                           int repeats);

// Ordinary least-squares fit of time = intercept + slope * n.
LinearFit fit_line(const std::vector<long long>& sizes,
                   const std::vector<double>& times);

// Solve cpu(n) == gpu(n) for n. Returns has_crossover=false when the
// lines are parallel, the crossing is negative, or the data was bad.
Prediction predict_crossover(const LinearFit& cpu_fit,
                             const LinearFit& gpu_fit,
                             long long min_n,
                             long long max_n);

// Geometric sampling search that never uses the linear formula.
// Starts at start_n, multiplies by growth_factor, stops at max_n or
// when the measurement budget runs out, then bisects the first
// bracket and verifies the candidate with fresh measurements.
SearchResult adaptive_search(CustomWorkload& work,
                             TimingMode mode,
                             long long start_n,
                             long long max_n,
                             double growth_factor,
                             int repeats,
                             int budget);

// Append-free CSV writer. Columns:
// workload,mode,n,cpu_ms,gpu_ms,repeats,valid
bool write_measurements_csv(const std::string& file_path,
                            const std::string& workload_name,
                            const std::string& mode_name,
                            const std::vector<Measurement>& rows);

}  // namespace cudagauge
