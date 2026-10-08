#pragma once
#include <algorithm>
#include <atomic>

#include "thread_pool.hpp"

namespace tp {

// Run f(i) over [begin, end), helping the pool until all chunks finish.
// The default grain creates roughly eight chunks per worker. f must not throw.
template <class F>
void parallel_for(thread_pool& pool, long begin, long end, F f, long grain = 0) {
    if (begin >= end) return;
    if (grain <= 0)
        grain = std::max<long>(1, (end - begin) / (8L * pool.size()));

    const long chunks = (end - begin + grain - 1) / grain;
    std::atomic<long> remaining{chunks};
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const long chunk_begin = begin + chunk * grain;
        const long chunk_end = std::min(end, chunk_begin + grain);
        pool.submit_detached([&, chunk_begin, chunk_end] {
            for (long i = chunk_begin; i < chunk_end; ++i) f(i);
            --remaining;
        });
    }
    pool.help_while([&] { return remaining.load() == 0; });
}

}  // namespace tp
