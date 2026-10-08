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
#include <type_traits>
#include <utility>
#include <vector>

namespace tp {

// Workers pop from their own deque's back and steal from others' fronts.
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

    void shutdown() {
        {
            std::lock_guard lock(cv_mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& thread : threads_)
            if (thread.joinable()) thread.join();
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using result_type = std::invoke_result_t<F, Args...>;
        // std::function needs a copyable wrapper for the move-only packaged task.
        auto packaged = std::make_shared<std::packaged_task<result_type()>>(
            [f = std::forward<F>(f), ... args = std::forward<Args>(args)]() mutable {
                return f(args...);
            });
        auto future = packaged->get_future();
        enqueue([packaged] { (*packaged)(); });
        return future;
    }

    // Detached tasks must not throw.
    void submit_detached(std::function<void()> f) { enqueue(std::move(f)); }

    void wait_idle() {
        std::unique_lock lk(cv_mu_);
        idle_cv_.wait(lk, [&] { return pending_ == 0; });
    }

    // Run tasks while waiting so nested parallel work cannot deadlock the pool.
    template <class Pred>
    void help_while(Pred pred) {
        const int me = tls_pool_ == this ? tls_index_ : -1;
        while (!pred()) {
            if (auto t = try_get(me)) run(std::move(t));
            else std::this_thread::yield();
        }
    }

    unsigned size() const { return static_cast<unsigned>(workers_.size()); }
    static thread_pool* current() { return tls_pool_; }
    static int current_worker() { return tls_pool_ ? tls_index_ : -1; }

private:
    using task = std::function<void()>;

    // Keep workers on separate cache lines.
    struct alignas(64) worker {
        std::mutex mutex;
        std::deque<task> queue;
        std::atomic<long> queued{0};
    };

    std::vector<worker> workers_;
    std::vector<std::thread> threads_;
    bool stealing_;
    std::atomic<unsigned> next_worker_{0};

    std::mutex cv_mu_;
    std::condition_variable cv_, idle_cv_;
    bool stop_ = false;  // guarded by cv_mu_
    std::atomic<int> sleepers_{0};
    std::atomic<int> spinning_{0};
    // Isolate the shared task count from the sleep/wake state.
    alignas(64) std::atomic<long> pending_{0};

    inline static thread_local thread_pool* tls_pool_ = nullptr;
    inline static thread_local int tls_index_ = -1;

    void enqueue(task t) {
        {
            std::lock_guard lk(cv_mu_);
            if (stop_) throw std::runtime_error("thread_pool: submit after shutdown");
        }
        const unsigned i = tls_pool_ == this ? static_cast<unsigned>(tls_index_)
                                             : next_worker_++ % workers_.size();
        auto& worker = workers_[i];
        pending_++;
        // Publish the count first so a worker preparing to sleep sees the task.
        worker.queued++;
        {
            std::lock_guard lock(worker.mutex);
            worker.queue.push_back(std::move(t));
        }
        // A scanning worker will find the task if stealing is enabled.
        // Sequentially consistent counts and the locked sleep check prevent lost wakeups.
        if (sleepers_.load() > 0 && (!stealing_ || spinning_.load() == 0)) {
            std::lock_guard lk(cv_mu_);
            // Without stealing, notify_all ensures the task's owner wakes up.
            if (stealing_) cv_.notify_one(); else cv_.notify_all();
        }
    }

    static task pop_back(worker& w) {
        std::lock_guard lock(w.mutex);
        if (w.queue.empty()) return {};
        task t = std::move(w.queue.back());
        w.queue.pop_back();
        w.queued--;
        return t;
    }
    static task steal_front(worker& w) {
        std::unique_lock lock(w.mutex, std::try_to_lock);
        if (!lock || w.queue.empty()) return {};
        task t = std::move(w.queue.front());
        w.queue.pop_front();
        w.queued--;
        return t;
    }

    task try_get(int me) {
        if (me >= 0) {
            if (auto t = pop_back(workers_[static_cast<size_t>(me)])) return t;
            if (!stealing_) return {};
        }

        const unsigned n = size();
        const unsigned start = me >= 0 ? static_cast<unsigned>(me) + 1 : next_worker_.load();
        for (unsigned k = 0; k < n; ++k) {
            const unsigned victim = (start + k) % n;
            if (static_cast<int>(victim) == me) continue;
            if (auto t = steal_front(workers_[victim])) return t;
        }
        return {};
    }

    bool work_available(int me) {
        if (!stealing_) return workers_[static_cast<size_t>(me)].queued > 0;
        for (auto& w : workers_)
            if (w.queued > 0) return true;
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
            // Briefly spin to avoid a sleep/wake cycle for short gaps in work.
            task t;
            spinning_++;
            for (int spin = 0; spin < 64 && !t; ++spin) t = try_get(m);
            spinning_--;
            if (t) {
                run(std::move(t));
                continue;
            }

            std::unique_lock lk(cv_mu_);
            sleepers_++;
            cv_.wait(lk, [&] { return stop_ || work_available(m); });
            sleepers_--;
            if (stop_ && !work_available(m)) break;
        }
    }
};

}  // namespace tp
