# taskpool — work-stealing thread pool + task scheduler (C++20)

Header-only, no dependencies, ~310 lines of library code.

| file | what |
|---|---|
| `include/tp/thread_pool.hpp` | `Job` / `ReturnJob<T>`, per-worker deques, work stealing, `push()` and `submit()` |
| `include/tp/parallel_for.hpp` | data-parallel loop with grain-size chunking |
| `include/tp/task_graph.hpp` | DAG task dependencies via atomic countdown |
| `tests/tests.cpp` | 16 tests (no framework), sanitizer-clean |
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

The pool also accepts explicit jobs using unique ownership:

```cpp
struct SquareJob : tp::ReturnJob<int> {
    int x;
    explicit SquareJob(int value) : x(value) {}
    int do_calc() override { return x * x; }
};

auto job = std::make_unique<SquareJob>(7);
auto result = job->get_future();
pool.push(std::move(job));
result.get(); // 49
```

`ReturnJob<void>` supports jobs without a result. `push(nullptr)` throws
`std::invalid_argument`. Retrieve a job's future once, before moving the job
into the pool.

## Build & run

```bash
make test    # build + run tests   (g++ >= 11 or clang >= 14, -std=c++20)
make tsan    # tests under ThreadSanitizer
make asan    # tests under AddressSanitizer + UBSan
make bench   # benchmarks (runtime depends on hardware and scheduling overhead)
```

## Design

The pool follows the explicit-job design:

- Each worker has a mutex and a deque of `unique_ptr<Job>`. It pops from the
  back and steals from other deques' fronts using `try_lock`.
- `push()` inserts a job and wakes a worker. External jobs use round robin;
  jobs spawned by a worker stay on its own deque.
- `pending_` counts queued jobs for the sleep predicate. `unfinished_` also
  includes running jobs, so `wait_idle()` waits for actual completion.
- Shutdown rejects new jobs, drains accepted work, and joins workers.
  Finish graphs and recursive spawners first because they submit more work.
- `ReturnJob<T>` uses your promise-based interface. The optional callable API
  has one small `FunctionJob` adapter; `submit()` uses `std::packaged_task`
  to return values and exceptions without another custom wrapper.

The graph and `parallel_for` code are unchanged. Both use `help_while()` to run
queued jobs during a nested wait. `wait_idle()` and shutdown are for callers
outside pool tasks; ordinary `future::get()` does not help execute jobs.

Move-only callables and arguments are supported; use `std::ref` for reference
arguments. Plain-job exceptions are caught so workers survive. Graph node
functions and `parallel_for` bodies must return normally for their completion
counts to advance.

With stealing disabled, external submissions wake all workers so the owner
wakes; local submissions need no notification. Explicit helpers can still take
jobs from any queue. Submission uses the wait mutex to coordinate with sleeping
and shutdown; separate deque locks protect removal. Shutdown calls must be
serialized, and callers must stop accessing the object before destruction.

## Benchmarks

Measured on October 9, 2026, on a Core i7-12650H reported by WSL2 as 16 hardware
threads (8 cores + HT), using GCC 11.4.0 with `-std=c++20 -O2 -pthread`.
Raw output: [run_09_10_26_wsl2_windows.txt](run_09_10_26_wsl2_windows.txt).
Numbers vary by machine and run; run `make bench` for your own. The task-body
labels are synthetic loop sizes, not calibrated wall-clock durations.

**A1. Aggregate throughput — tasks spawned inside workers.** Each worker
spawns its own share of tasks (worker-local push/pop path plus stealing).
Counters are per-worker and cache-line padded so the benchmark measures the
pool, not a shared atomic.

| task body | 1 thread | 2 | 4 | 8 | 16 | scaling @16 |
|---|---|---|---|---|---|---|
| ~0 µs (pure overhead) | 10.06M/s · **99 ns/task** | 4.84M/s | 3.78M/s | 3.07M/s | 1.78M/s | 0.18× |
| ~1 µs | 1.23M/s | 1.88M/s | 2.58M/s | 2.74M/s | 2.26M/s | 1.84× |
| ~10 µs | 131k/s | 255k/s | 478k/s | 748k/s | 842k/s | **6.41×** |

The 0 µs row measures scheduling overhead; larger task bodies amortize it.

**A2. Single external producer.** Main thread submitting one task at a time
to 16 workers — bounded by the submitter, so this is "cost of an external
submit", not a scaling number.

| | tasks/sec |
|---|---|
| `thread_pool::submit_detached` | **91.8k/s** (10,890 ns/task) |
| `std::async` (thread per task) | 25.8k/s |

The [August implementation](run_17_08_26_wsl2_windows.txt) recorded 1.87M/s
in a separate run. The simpler wakeup path costs throughput for tiny jobs.

**B. Work stealing under skewed load.** 20,000 tasks of ~5 µs, all spawned
from inside one worker, so every task lands in one deque. Start latency is
the time from enqueue to the task's first instruction. The benchmark runs
two repetitions; the table uses the first.

| | makespan | start-latency p50 | p99 |
|---|---|---|---|
| stealing OFF | **73.8 ms** | 36.63 ms | 72.95 ms |
| stealing ON | 218.3 ms | **0.01 ms** | **0.02 ms** |

Without stealing the load runs serially on its owner. With stealing, other
workers drain its deque while the owner submits work. Submission overhead
and waking workers still contribute to the measured makespan. In this tiny-job
workload, those costs outweigh the benefit of load balancing: stealing improves
start latency but makes total runtime worse. The second repetition measured
74.1 ms with stealing off and 215.8 ms with stealing on.

**C. `parallel_for` — sum of sin(i), N = 8M, compute-bound.**

| | time | speedup |
|---|---|---|
| sequential | 66.0 ms | — |
| `parallel_for`, grain = auto (N / 8·threads) | **7.4 ms** | **8.95×** |
| grain = 4096 | 25.6 ms | 2.57× |
| grain = N/2 (2 tasks: no load balance) | 31.3 ms | 2.11× |
| grain = 16 (500k tasks: overhead dominates) | 5,367.8 ms | 0.01× |

## Tests

`make test` runs 16 tests covering explicit jobs and callable futures,
exceptions, move-only arguments, idle waiting, shutdown races, stealing,
graph dependencies, nested loops and graphs, and recursive spawning.

All 16 tests passed on native Windows (MinGW GCC 15.2.0) and WSL2 (GCC 11.4.0).
The WSL2 runs also passed ThreadSanitizer and AddressSanitizer + UBSan without
diagnostics.

## Limitations / future work

* Deques use mutexes.
* Shared queue accounting and a wait mutex on submission and idle notification.
* Graph node functions and `parallel_for` bodies must not throw; their completion
  bookkeeping requires normal return even though the pool catches job exceptions.
* No task priorities.
