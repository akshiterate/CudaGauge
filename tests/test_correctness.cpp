// Level 1 test: every built-in workload must produce matching CPU
// and GPU results in both timing modes at small sizes.

#include <cstdio>

#include "cudagauge/cudagauge.h"
#include "cudagauge/workloads.h"

using namespace cudagauge;

static int failures = 0;

static void check_workload(CustomWorkload (*make)(), long long n, const char* label) {
    TimingMode modes[2] = {TimingMode::EndToEnd, TimingMode::Resident};
    for (int m = 0; m < 2; m++) {
        CustomWorkload work = make();
        Measurement got = run_custom_size(work, n, modes[m], 2);
        bool pass = got.valid;
        std::printf("%-14s n=%-8lld %-8s : %s (cpu %.4f ms, gpu %.4f ms)\n", label, n,
                    mode_to_string(modes[m]).c_str(), pass ? "PASS" : "FAIL", got.cpu_ms,
                    got.gpu_ms);
        if (!pass) {
            failures++;
        }
    }
}

int main() {
    check_workload(make_vecadd_workload, 1024, "vecadd");
    check_workload(make_vecadd_workload, 100003, "vecadd-odd");
    check_workload(make_reduce_workload, 1024, "reduce");
    check_workload(make_reduce_workload, 100003, "reduce-odd");
    check_workload(make_matmul_workload, 16, "matmul");
    check_workload(make_matmul_workload, 37, "matmul-odd");
    if (failures > 0) {
        std::printf("%d FAILURES\n", failures);
        return 1;
    }
    std::printf("all correctness tests passed\n");
    return 0;
}
