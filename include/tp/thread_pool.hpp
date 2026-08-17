#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tp {

// Work-stealing thread pool.
//
// THE WHOLE IDEA IN 4 LINES:
//   1. Every worker has its OWN deque of tasks, guarded by its OWN mutex.
//   2. A worker takes from the BACK of its own deque (LIFO: cache-warm work).
//   3. If its deque is empty, it STEALS from the FRONT of another worker's
//      deque (FIFO: the oldest / biggest chunk), using try_lock so it never
//      blocks behind a busy owner.
//   4. Idle workers sleep on a condition variable; submit() wakes one.
//
// Why not one shared queue?  Every thread would fight over one lock and one
// cache line. Per-worker deques spread that contention N ways, and in steady
// state each lock is touched only by its owner.
class thread_pool {
public:
    explicit thread_pool(unsigned n = std::thread::hardware_concurrency(),
                         bool stealing = true)
        : workers_(n ? n : 1), stealing_(stealing) {
        for (unsigned i = 0; i < workers_.size(); ++i)
            threads_.emplace_back([this, i] { worker_loop(i); });
    }
    ~thread_pool() { shutdown(); }
    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    // Stop accepting work, finish everything already queued, join.
    void shutdown() {
        { std::lock_guard lk(cv_mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : threads_) if (t.joinable()) t.join();
    }

    // submit(f, args...) -> std::future<result>. Exceptions thrown by f are
    // rethrown from future::get() (std::packaged_task does this for us).
    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;
        // packaged_task is move-only, std::function needs copyable:
        // wrap it in a shared_ptr. (Classic thread-pool interview question.)
        auto pt = std::make_shared<std::packaged_task<R()>>(
            [f = std::forward<F>(f), ... args = std::forward<Args>(args)]() mutable {
                return f(args...);
            });
        auto fut = pt->get_future();
        enqueue([pt] { (*pt)(); });
        return fut;
    }

    // Fire-and-forget: no future, cheaper. f must not throw.
    void submit_detached(std::function<void()> f) { enqueue(std::move(f)); }

    // Block until every submitted task has finished.
    void wait_idle() {
        std::unique_lock lk(cv_mu_);
        idle_cv_.wait(lk, [&] { return pending_ == 0; });
    }

    // Run queued tasks on THIS thread until pred() is true. This is how a
    // task can wait for its own subtasks without deadlocking the pool: the
    // waiting worker keeps working instead of blocking.
    template <class Pred>
    void help_while(Pred pred) {
        int me = (tls_pool_ == this) ? tls_index_ : -1;
        while (!pred()) {
            if (auto t = try_get(me)) run(std::move(t));
            else std::this_thread::yield();
        }
    }

    unsigned size() const { return static_cast<unsigned>(workers_.size()); }
    static thread_pool* current() { return tls_pool_; }        // nullptr if not a worker
    static int current_worker() { return tls_pool_ ? tls_index_ : -1; }

private:
    using task = std::function<void()>;

    struct alignas(64) worker {   // alignas: no false sharing between workers
        std::mutex mu;
        std::deque<task> q;       // owner pops BACK, thieves pop FRONT
        std::atomic<long> size{0};  // q.size(), readable without the mutex
    };

    std::vector<worker> workers_;
    std::vector<std::thread> threads_;
    bool stealing_;
    std::atomic<unsigned> rr_{0};       // round-robin target for outside submits

    std::mutex cv_mu_;
    std::condition_variable cv_, idle_cv_;
    bool stop_ = false;                 // guarded by cv_mu_
    std::atomic<int> sleepers_{0};      // workers currently blocked on cv_
    std::atomic<int> spinning_{0};      // workers awake and scanning for work
    // The one global per-task counter. It is the throughput ceiling for
    // near-empty tasks (two contended atomic RMWs per task from every
    // core); it lives on its own cache line so it at least doesn't drag
    // the mutex or sleepers_ along. Per-deque counts live in worker::size.
    alignas(64) std::atomic<long> pending_{0};   // submitted, not yet finished

    inline static thread_local thread_pool* tls_pool_ = nullptr;
    inline static thread_local int tls_index_ = -1;

    void enqueue(task t) {
        {
            std::lock_guard lk(cv_mu_);
            if (stop_) throw std::runtime_error("thread_pool: submit after shutdown");
        }
        // From a worker: push to its OWN deque. From outside: round-robin.
        unsigned i = (tls_pool_ == this) ? static_cast<unsigned>(tls_index_)
                                         : rr_++ % workers_.size();
        pending_++;
        workers_[i].size++;   // before the push: the count may briefly exceed
                              // the deque, never undercount it
        {
            std::lock_guard lk(workers_[i].mu);
            workers_[i].q.push_back(std::move(t));
        }
        // Wake a worker only if (a) some worker is actually asleep AND
        // (b) no worker is already awake and scanning -- a spinner will
        // find this task on its own, and waking more threads per submit
        // just creates a herd fighting over one mutex (measured: ~17 us per
        // submit on a 16-thread machine without this check).
        // Correctness: sleepers_++ (worker) and size++ (here) are both
        // seq_cst, so at least one side sees the other's write -- either we
        // see a sleeper and notify (under cv_mu_, so it is provably asleep),
        // or the worker sees size > 0 and doesn't sleep. A spinner
        // re-checks the sizes under cv_mu_ before it may sleep, so skipping
        // the notify because of it can't strand the task.
        // With stealing OFF, a spinner can't take another worker's task, so
        // the spinner shortcut doesn't apply there.
        if (sleepers_.load() > 0 && (!stealing_ || spinning_.load() == 0)) {
            std::lock_guard lk(cv_mu_);
            // With stealing ON, any woken worker can take the task, so one
            // wakeup is enough. With stealing OFF only the target worker
            // can run it -- notify_one might wake the wrong worker, which
            // would recheck its own (empty) deque and sleep again while the
            // right one stays asleep forever. So wake them all.
            if (stealing_) cv_.notify_one(); else cv_.notify_all();
        }
    }

    static task pop_back(worker& w) {
        std::lock_guard lk(w.mu);
        if (w.q.empty()) return nullptr;
        task t = std::move(w.q.back());
        w.q.pop_back();
        w.size--;
        return t;
    }
    static task steal_front(worker& w) {
        std::unique_lock lk(w.mu, std::try_to_lock);   // busy? move on, don't wait
        if (!lk || w.q.empty()) return nullptr;
        task t = std::move(w.q.front());
        w.q.pop_front();
        w.size--;
        return t;
    }

    // Own deque first, then steal from the others (starting at a neighbour).
    task try_get(int me) {
        task t;
        if (me >= 0) t = pop_back(workers_[static_cast<size_t>(me)]);
        if (!t && (stealing_ || me < 0)) {
            unsigned n = static_cast<unsigned>(workers_.size());
            unsigned start = me >= 0 ? static_cast<unsigned>(me) + 1 : rr_.load();
            for (unsigned k = 0; k < n && !t; ++k) {
                unsigned v = (start + k) % n;
                if (static_cast<int>(v) != me) t = steal_front(workers_[v]);
            }
        }
        return t;
    }

    // Sleep-predicate check: with stealing, any deque counts; without it,
    // only my own. Only runs on the (rare) about-to-sleep path.
    bool work_available(int me) {
        if (!stealing_) return workers_[static_cast<size_t>(me)].size > 0;
        for (auto& w : workers_) if (w.size > 0) return true;
        return false;
    }

    void run(task t) {
        t();
        if (--pending_ == 0) {
            std::lock_guard lk(cv_mu_);
            idle_cv_.notify_all();
        }
    }

    void worker_loop(unsigned me) {
        tls_pool_ = this;
        tls_index_ = static_cast<int>(me);
        const int m = static_cast<int>(me);
        for (;;) {
            // Spin briefly before sleeping: under steady load the next task
            // arrives within microseconds, and a futex sleep/wake costs far
            // more than a few failed scans.
            task t;
            spinning_++;
            for (int spin = 0; spin < 64 && !t; ++spin) t = try_get(m);
            spinning_--;
            if (t) { run(std::move(t)); continue; }

            std::unique_lock lk(cv_mu_);
            sleepers_++;
            cv_.wait(lk, [&] { return stop_ || work_available(m); });
            sleepers_--;
            if (stop_ && !work_available(m)) break;   // drained
        }
    }
};

}  // namespace tp
