// 10 tests, no framework. Run with:  make test   (also: make tsan / make asan)
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "tp/parallel_for.hpp"
#include "tp/task_graph.hpp"
#include "tp/thread_pool.hpp"

#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)
#define RUN(fn)  do { std::printf("[ RUN ] %s\n", #fn); fn(); std::printf("[ OK  ] %s\n", #fn); } while (0)

static void submit_returns_values() {
    tp::thread_pool pool(4);
    auto a = pool.submit([] { return 42; });
    auto b = pool.submit([](int x, int y) { return x + y; }, 20, 22);
    auto c = pool.submit([] { return std::string("hi"); });
    CHECK(a.get() == 42 && b.get() == 42 && c.get() == "hi");
}

static void exception_comes_back_through_future() {
    tp::thread_pool pool(2);
    auto f = pool.submit([]() -> int { throw std::runtime_error("boom"); });
    bool caught = false;
    try { f.get(); } catch (const std::runtime_error&) { caught = true; }
    CHECK(caught);
}

static void detached_tasks_all_run() {
    std::atomic<int> n{0};
    tp::thread_pool pool(8);
    for (int i = 0; i < 100'000; ++i) pool.submit_detached([&] { ++n; });
    pool.wait_idle();
    CHECK(n == 100'000);
}

// Tasks spawning tasks (binary tree, 2^16 leaves): exercises the
// worker-local push + stealing.
static void recursive_spawn_tree() {
    std::atomic<long> leaves{0};
    tp::thread_pool pool(8);
    std::function<void(int)> go = [&](int d) {
        if (d == 0) { ++leaves; return; }
        pool.submit_detached([&, d] { go(d - 1); });
        pool.submit_detached([&, d] { go(d - 1); });
    };
    pool.submit_detached([&] { go(16); });
    pool.wait_idle();
    CHECK(leaves == (1L << 16));
}

static void destructor_drains_queued_work() {
    std::atomic<int> ran{0};
    {
        tp::thread_pool pool(2);
        for (int i = 0; i < 500; ++i)
            pool.submit_detached([&] {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                ++ran;
            });
    }  // destructor must finish all 500 before joining
    CHECK(ran == 500);
}

static void stealing_disabled_still_correct() {
    tp::thread_pool pool(4, /*stealing=*/false);
    std::atomic<int> n{0};
    pool.submit([&] {   // spawn from inside one worker -> all in one deque
        for (int i = 0; i < 10'000; ++i) pool.submit_detached([&] { ++n; });
    }).get();
    pool.wait_idle();
    CHECK(n == 10'000);

    // Regression: let every worker fall asleep, THEN submit from outside.
    // notify_one could wake a worker whose own deque is empty; with stealing
    // off it would go back to sleep and the target worker would never wake.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (int i = 0; i < 8; ++i) {
        auto f = pool.submit([] { return 1; });
        CHECK(f.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // let them sleep again
    }
}

static void dag_respects_dependencies() {
    tp::thread_pool pool(4);
    // Repeat: the source-scan/exec race only shows up on some interleavings.
    for (int rep = 0; rep < 2000; ++rep) {
        std::mutex mu;
        std::string order;
        auto rec = [&](char c) { std::lock_guard lk(mu); order += c; };

        tp::task_graph g(pool);
        int a = g.add([&] { rec('a'); });
        int b = g.add([&] { rec('b'); }, {a});
        int c = g.add([&] { rec('c'); }, {a});
        g.add([&] { rec('d'); }, {b, c});
        g.run_and_wait();
        CHECK(order.size() == 4 && order.front() == 'a' && order.back() == 'd');
    }

    // Fan-out / fan-in: sink must observe all 500 middles finished.
    tp::task_graph g2(pool);
    int src = g2.add([] {});
    std::atomic<int> mids{0};
    std::vector<int> layer;
    for (int i = 0; i < 500; ++i) layer.push_back(g2.add([&] { ++mids; }, {src}));
    int seen = -1;
    g2.add([&] { seen = mids; }, layer);
    g2.run_and_wait();
    CHECK(seen == 500);
}

static void parallel_for_matches_sequential() {
    tp::thread_pool pool(8);
    const long N = 1'000'000;
    std::vector<long> v(N);
    tp::parallel_for(pool, 0, N, [&](long i) { v[static_cast<size_t>(i)] = i * i; });
    for (long i = 0; i < N; i += 997) CHECK(v[static_cast<size_t>(i)] == i * i);
    std::atomic<int> n{0};
    tp::parallel_for(pool, 0, 0, [&](long) { ++n; });     // empty range
    tp::parallel_for(pool, 0, 3, [&](long) { ++n; }, 100); // grain > range
    CHECK(n == 3);
}

// parallel_for called from INSIDE a task on a 1-thread pool: the only
// worker must help (run the chunks itself) instead of blocking -> no deadlock.
static void nested_parallel_for_does_not_deadlock() {
    tp::thread_pool pool(1);
    std::atomic<long> sum{0};
    pool.submit([&] {
        tp::parallel_for(pool, 0, 10'000, [&](long i) { sum += i; });
    }).get();
    CHECK(sum == 9999L * 10'000 / 2);
}

static void stress_random_spawns() {
    std::atomic<long> work{0};
    tp::thread_pool pool;
    std::mt19937 rng(123);
    for (int round = 0; round < 3; ++round) {
        long expected = 0;
        for (int i = 0; i < 2000; ++i) {
            int fan = static_cast<int>(rng() % 4);
            expected += 1 + fan;
            pool.submit_detached([&, fan] {
                ++work;
                for (int c = 0; c < fan; ++c) pool.submit_detached([&] { ++work; });
            });
        }
        pool.wait_idle();
        CHECK(work.exchange(0) == expected);
    }
}

int main() {
    RUN(submit_returns_values);
    RUN(exception_comes_back_through_future);
    RUN(detached_tasks_all_run);
    RUN(recursive_spawn_tree);
    RUN(destructor_drains_queued_work);
    RUN(stealing_disabled_still_correct);
    RUN(dag_respects_dependencies);
    RUN(parallel_for_matches_sequential);
    RUN(nested_parallel_for_does_not_deadlock);
    RUN(stress_random_spawns);
    std::printf("\nAll tests passed.\n");
}
