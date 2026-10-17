// Unit test for the adaptive search using a synthetic workload with
// no GPU code: the CPU side busy-loops proportional to n, the GPU side
// busy-loops with a fixed cost plus a smaller slope, so a real
// crossover exists near n = 10000 and the search must find a bracket.

#include <cstdio>

#include "cudagauge/cudagauge.h"

using namespace cudagauge;

static long long current_n = 0;

// Burn roughly n * K simple operations so timing scales with n.
static void burn(long long ops) {
    volatile double acc = 0.0;
    for (long long i = 0; i < ops; i++) {
        acc += (double)(i % 7) * 0.5;
    }
    (void)acc;
}

int main() {
    CustomWorkload fake;
    fake.name = "fake";
    fake.setup = [](long long n) { current_n = n; };
    fake.cpu_work = []() { burn(current_n * 5000); };
    fake.gpu_e2e_work = []() { burn(50000000 + current_n * 50); };
    fake.gpu_kernel_work = fake.gpu_e2e_work;
    fake.teardown = []() {};
    fake.validate = []() { return true; };

    SearchResult found = adaptive_search(fake, TimingMode::EndToEnd, 100, 1000000, 4.0, 3, 16);
    std::printf("found=%d crossover=%lld bracket=[%lld,%lld] measures=%d note=%s\n",
                (int)found.found, found.crossover_n, found.lower_n, found.upper_n,
                found.num_measurements, found.note.c_str());
    if (!found.found) {
        std::printf("FAIL search should find the synthetic crossover\n");
        return 1;
    }
    if (found.lower_n >= found.upper_n || found.upper_n <= 0) {
        std::printf("FAIL bracket is not a real interval\n");
        return 1;
    }
    std::printf("all search tests passed\n");
    return 0;
}
