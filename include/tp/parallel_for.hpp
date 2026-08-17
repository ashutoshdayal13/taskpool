#pragma once
#include <algorithm>
#include <atomic>

#include "thread_pool.hpp"

namespace tp {

// parallel_for(pool, begin, end, f): calls f(i) for i in [begin, end) across
// the pool. Blocks until done.
//
//   * The range is cut into chunks of `grain` iterations, one task each.
//     Default grain = n / (8 * threads): ~8 chunks per worker -- enough
//     slack for stealing to fix imbalance, few enough that per-task
//     overhead stays negligible. (Too small: overhead dominates. Too big:
//     no load balance. bench.cpp shows both.)
//   * The caller waits by HELPING (running queued tasks) rather than
//     blocking, so calling parallel_for from inside a task can't deadlock.
template <class F>
void parallel_for(thread_pool& pool, long begin, long end, F f, long grain = 0) {
    if (begin >= end) return;
    if (grain <= 0)
        grain = std::max<long>(1, (end - begin) / (8L * pool.size()));

    long nchunks = (end - begin + grain - 1) / grain;
    std::atomic<long> remaining{nchunks};
    for (long c = 0; c < nchunks; ++c) {
        long b = begin + c * grain, e = std::min(end, b + grain);
        pool.submit_detached([&, b, e] {
            for (long i = b; i < e; ++i) f(i);
            --remaining;
        });
    }
    pool.help_while([&] { return remaining.load() == 0; });
}

}  // namespace tp
