# taskpool — work-stealing thread pool + task scheduler (C++20)

Header-only, no dependencies, ~300 lines of library code.

| file | what |
|---|---|
| `include/tp/thread_pool.hpp` | workers, per-worker deques, work stealing, sleep/wake, `submit()` → future |
| `include/tp/parallel_for.hpp` | data-parallel loop with grain-size chunking |
| `include/tp/task_graph.hpp` | DAG task dependencies via atomic countdown |
| `tests/tests.cpp` | 10 tests (no framework), sanitizer-clean |
| `bench/bench.cpp` | throughput / stealing / `parallel_for` benchmarks |

## Usage

```cpp
#include "tp/thread_pool.hpp"
#include "tp/parallel_for.hpp"
#include "tp/task_graph.hpp"

tp::thread_pool pool;                                   // hardware_concurrency workers

auto f = pool.submit([](int x) { return x * 2; }, 21);  // -> std::future<int>
f.get();                                                // 42; exceptions rethrow here
pool.submit_detached([] { /* fire and forget */ });
pool.wait_idle();                                       // block until everything finished

tp::parallel_for(pool, 0, n, [&](long i) { out[i] = fn(in[i]); });

tp::task_graph g(pool);
int a = g.add(load);
int b = g.add(transform, {a});      // runs after a
int c = g.add(validate,  {a});
g.add(publish, {b, c});             // runs after b AND c
g.run_and_wait();
```

Calling `parallel_for` or `task_graph::run_and_wait` from *inside* a pool task
is safe: the caller helps run queued tasks instead of blocking, so nested
parallelism cannot deadlock the pool.

## Build & run

```bash
make test    # build + run tests   (g++ >= 11 or clang >= 14, -std=c++20)
make tsan    # tests under ThreadSanitizer
make asan    # tests under AddressSanitizer + UBSan
make bench   # benchmarks (~30 s)
```

## Design

**Per-worker deques, not one shared queue.** Each worker owns a `std::deque`
guarded by its own mutex. The owner pushes and pops at the *back* (LIFO — the
task it just spawned is still hot in cache). When its deque is empty it
*steals* from the *front* of another worker's deque (FIFO — the oldest and,
in divide-and-conquer workloads, largest chunk of work). Stealing uses
`try_lock`: a busy victim is skipped, never waited on, so thieves don't queue
up behind a busy owner. Tasks submitted from a worker go straight into that
worker's own deque; tasks submitted from outside are round-robined.

A single shared queue would make every thread contend on one lock and one
cache line. N per-worker deques spread that contention N ways, and in steady
state each lock is only ever touched by its owner. Each `worker` struct is
`alignas(64)` so two workers' mutexes never share a cache line.

**Sleeping without lost wakeups, and without a herd.** A worker that finds
nothing spins through ~64 scans (a futex sleep/wake costs far more), then
waits on a condition variable with predicate "stop, or some deque is
non-empty". `submit()` bumps the target deque's atomic size counter, then
notifies only if some worker is asleep *and* no worker is already awake and
scanning (`spinning_ == 0`) — a spinner will find the task on its own, and
waking a thread per submit creates a herd contending on one mutex. The
size and sleeper counters are seq_cst, so at least one side always sees the
other's write: either the submitter sees a sleeper and notifies (under the cv
mutex, so the sleeper is provably asleep), or the worker sees work and doesn't
sleep. In steady state `submit()` never touches the global mutex.

**Waiting inside a task.** If a task blocks on its own subtasks and every
worker does the same, nothing runs. `help_while(pred)` runs queued tasks on
the *waiting* thread until `pred` holds; `parallel_for` and `task_graph` wait
this way. The test suite runs a nested `parallel_for` on a 1-thread pool to
prove it.

**Futures.** `submit()` wraps the callable in a `std::packaged_task`; its
future is returned to the caller. `packaged_task` is move-only and
`std::function` requires copyable, so the task is held via `shared_ptr`.

**DAG.** Each node keeps an atomic count of unfinished predecessors. When a
node finishes it decrements each successor's count; whoever hits zero submits
that successor. Edges are frozen before `run_and_wait()`, so no locks at
runtime. Sources are collected *before* any is submitted (submitting while
scanning could double-submit a successor that an already-running source has
released).

**`parallel_for`.** The range is cut into chunks of `grain` iterations, one
task each. Default grain is `n / (8 × threads)` — about 8 chunks per worker,
enough slack for stealing to fix imbalance without drowning in per-task
overhead. Too small and overhead dominates; too big and there is no load
balancing. Both ends are shown in the benchmark below.

**Shutdown.** The destructor sets `stop_`, wakes everyone, and workers exit
only once their work is drained, so accepted tasks are never dropped.
`submit()` after `shutdown()` throws.

**Known ceiling.** Every task increments and decrements one global atomic
(`pending_`, backing `wait_idle`). Under 16-way contention that is ~100+ ns of
serialized work per task, which caps near-empty-task throughput at a few
million tasks/sec regardless of thread count. Tasks of a few µs amortize it;
removing it would need per-worker counters plus an epoch scheme for
`wait_idle`.

## Benchmarks

Measured on a 16-hardware-thread machine (8 physical cores + HT), Linux under
WSL2, `g++ -O2`. Numbers vary by machine; run `make bench` for your own.

**A1. Aggregate throughput — tasks spawned inside workers.** Each worker
spawns its own share of tasks (worker-local push/pop path plus stealing).
Counters are per-worker and cache-line padded so the benchmark measures the
pool, not a shared atomic.

| task body | 1 thread | 2 | 4 | 8 | 16 | scaling @16 |
|---|---|---|---|---|---|---|
| ~0 µs (pure overhead) | 11.3M/s · **88 ns/task** | 6.6M/s | 5.8M/s | 5.1M/s | 3.2M/s | 0.28× |
| ~1 µs | 1.39M/s | 2.37M/s | 3.33M/s | 4.11M/s | 4.27M/s | 3.1× |
| ~10 µs | 148k/s | 281k/s | 538k/s | 928k/s | 1.47M/s | **9.9×** |

The 0 µs row is the scheduling overhead floor and shows the global-counter
ceiling described above; the 10 µs row is what realistic small tasks look
like (≈10× on 8 physical cores with hyperthreading).

**A2. Single external producer.** Main thread submitting one task at a time
to 16 workers — bounded by the submitter, so this is "cost of an external
submit", not a scaling number.

| | tasks/sec |
|---|---|
| `thread_pool::submit_detached` | **2.07M/s** (484 ns/task) |
| `std::async` (thread per task) | 29k/s |

**B. Work stealing under skewed load.** 20,000 tasks of ~5 µs, all spawned
from inside worker 0, so every task lands in one deque. Start latency is the
time from enqueue to the task's first instruction.

| | makespan | start-latency p50 | p99 |
|---|---|---|---|
| stealing OFF | 137 ms | 71 ms | 136 ms |
| stealing ON | **34 ms** | **0.00 ms** | **0.04 ms** |

Without stealing the load runs serially on worker 0. With stealing the other
workers drain its deque as fast as it fills; the remaining makespan is the
spawner's own submit loop.

**C. `parallel_for` — sum of sin(i), N = 8M, compute-bound.**

| | time | speedup |
|---|---|---|
| sequential | 51.8 ms | — |
| `parallel_for`, grain = auto (N / 8·threads) | **6.6 ms** | **7.8×** |
| grain = 4096 | 8.4 ms | 6.2× |
| grain = N/2 (2 tasks: no load balance) | 31.1 ms | 1.7× |
| grain = 16 (500k tasks: overhead dominates) | 339 ms | 0.15× |

## Tests

`make test` runs 10 tests: futures with values / arguments / exceptions,
100k detached tasks, a 2^16-leaf recursive spawn tree, destructor drains
queued work, submit-after-shutdown throws, stealing-disabled correctness plus
a wake-the-right-worker regression, DAG ordering (diamond, 500-wide fan-out/in,
repeated 2,000× to shake out interleavings), `parallel_for` correctness and
edge ranges, nested `parallel_for` on a 1-thread pool, and a randomized
spawn stress test. All pass under ThreadSanitizer and AddressSanitizer + UBSan.

Note: on distributions whose libstdc++ is not TSAN-instrumented (e.g. Ubuntu),
`make tsan` reports one known false positive on the exception-through-future
test — `std::exception_ptr`'s refcount lives inside libstdc++, so TSAN cannot
see the ordering. Nothing in this library is implicated.

## Limitations / future work

* Deques are mutex-guarded; a lock-free Chase-Lev deque would drop in behind
  the same push/pop/steal interface.
* One global `pending_` counter per task (see "Known ceiling").
* No task priorities.
