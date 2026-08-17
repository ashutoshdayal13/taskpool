// One benchmark file, three numbers for the CV:
//   A. tasks/sec vs thread count (+ std::async baseline)
//   B. work stealing ON vs OFF under a skewed load (makespan + p99 latency)
//   C. parallel_for speedup vs sequential (+ grain-size sweep)
// Usage: ./bench     (takes ~10-20 s)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

#include "tp/parallel_for.hpp"
#include "tp/thread_pool.hpp"

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
template <class T> static void keep(const T& v) { asm volatile("" : : "r,m"(v) : "memory"); }
static void spin_us(int us) {              // ~1 ns per iteration
    unsigned long x = 88172645463325252ull;
    for (int i = 0; i < us * 1000; ++i) x = x * 6364136223846793005ull + 1;
    keep(x);
}

int main() {
    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("hardware threads: %u\n", hw);

    // ---- A. throughput --------------------------------------------------
    // A1: aggregate throughput -- each worker spawns its own share of tasks
    //     (tasks spawning tasks: worker-local push/pop path + stealing).
    //     Counters are per-worker and cache-line padded: a single shared
    //     atomic in the task body would measure that atomic, not the pool.
    //     Two task sizes: near-empty (pure scheduler overhead, worst case)
    //     and ~1us of work (what real scaling looks like).
    struct alignas(64) counter { long v = 0; };
    const long N = 2'000'000;
    for (int body_us : {0, 1, 10}) {
        const long M = body_us >= 10 ? N / 10 : N;   // keep the 10us run short
        std::printf("\nA1. tasks spawned inside workers, task body ~%d us\n", body_us);
        double base = 0;
        for (unsigned t = 1; t <= hw; t *= 2) {
            std::vector<counter> cnt(t + 1);
            tp::thread_pool pool(t);
            auto t0 = clk::now();
            for (unsigned w = 0; w < t; ++w)                   // one seed per worker
                pool.submit_detached([&, t] {
                    for (long i = 0; i < M / t; ++i)
                        pool.submit_detached([&] {
                            if (body_us) spin_us(body_us);
                            cnt[static_cast<size_t>(tp::thread_pool::current_worker() + 1)].v++;
                        });
                });
            pool.wait_idle();
            double s = ms(t0, clk::now()) / 1e3;
            long total = 0; for (auto& c : cnt) total += c.v;
            if (t == 1) base = total / s;
            std::printf("   %2u threads: %10.0f tasks/sec  (%.0f ns/task)  %.2fx vs 1 thread\n",
                        t, total / s, 1e9 * s / total, (total / s) / base);
        }
    }
    // A2: one external producer -- bounded by main's submit rate, so this
    //     is "cost of an external submit", not a scaling number.
    std::printf("\nA2. single external producer (main thread submitting one task at a time)\n");
    {
        std::atomic<long> n{0};
        tp::thread_pool pool(hw);
        auto t0 = clk::now();
        for (long i = 0; i < N / 2; ++i) pool.submit_detached([&] { ++n; });
        pool.wait_idle();
        double s = ms(t0, clk::now()) / 1e3;
        std::printf("   %2u threads: %10.0f tasks/sec  (%.0f ns/task, submit-bound)\n", hw, n / s, 1e9 * s / n);
    }
    {
        const long M = 10'000;
        std::vector<std::future<long>> f;
        auto t0 = clk::now();
        for (long i = 0; i < M; ++i) f.push_back(std::async(std::launch::async, [i] { return i; }));
        long s = 0; for (auto& x : f) s += x.get(); keep(s);
        double sec = ms(t0, clk::now()) / 1e3;
        std::printf("   std::async (thread per task): %.0f tasks/sec\n", M / sec);
    }

    // ---- B. stealing on vs off ---------------------------------------------
    std::printf("\nB. work stealing, skewed load: 20000 x ~5us tasks, all spawned inside worker 0\n");
    for (int rep = 0; rep < 2; ++rep)
    for (bool stealing : {false, true}) {
        const long T = 20'000;
        tp::thread_pool pool(hw, stealing);
        std::vector<clk::time_point> enq(T);
        std::vector<double> lat(T);
        auto t0 = clk::now();
        pool.submit_detached([&] {
            for (long i = 0; i < T; ++i) {
                enq[i] = clk::now();
                pool.submit_detached([&, i] {
                    lat[i] = ms(enq[i], clk::now());
                    spin_us(5);
                });
            }
        });
        pool.wait_idle();
        double total = ms(t0, clk::now());
        std::sort(lat.begin(), lat.end());
        std::printf("   stealing %-3s  makespan %7.1f ms   start-latency p50 %7.2f ms  p99 %7.2f ms\n",
                    stealing ? "ON" : "OFF", total, lat[T / 2], lat[T * 99 / 100]);
    }

    // ---- C. parallel_for -----------------------------------------------------
    std::printf("\nC. parallel_for, sum of sin(i), N = 8M\n");
    const long P = 8'000'000;
    tp::thread_pool pool(hw);
    auto t0 = clk::now();
    double acc = 0;
    for (long i = 0; i < P; ++i) acc += std::sin(static_cast<double>(i));
    keep(acc);
    double seq = ms(t0, clk::now());
    std::printf("   sequential      %8.1f ms\n", seq);

    struct alignas(64) slot { double v = 0; };   // per-worker partial sums, no false sharing
    for (long grain : {0L, 16L, 4096L, P / 2}) {
        std::vector<slot> part(hw + 1);
        auto t1 = clk::now();
        tp::parallel_for(pool, 0, P, [&](long i) {
            part[static_cast<size_t>(tp::thread_pool::current_worker() + 1)].v += std::sin(static_cast<double>(i));
        }, grain);
        double par = ms(t1, clk::now());
        double tot = 0; for (auto& p : part) tot += p.v; keep(tot);
        std::printf("   parallel_for    %8.1f ms   speedup %5.2fx   grain=%s\n", par, seq / par,
                    grain == 0 ? "auto (N / 8*threads)" : grain == 16 ? "16 (500k tasks: overhead dominates)"
                    : grain == 4096 ? "4096" : "N/2 (2 tasks: no load balance)");
    }
}
