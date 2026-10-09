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

struct Job {
    virtual void execute() = 0;
    virtual ~Job() = default;
};

template <class T>
struct ReturnJob : Job {
    std::promise<T> pr;

    virtual T do_calc() = 0;

    void execute() override {
        try {
            if constexpr (std::is_void_v<T>) {
                do_calc();
                pr.set_value();
            } else {
                pr.set_value(do_calc());
            }
        } catch (...) {
            pr.set_exception(std::current_exception());
        }
    }

    std::future<T> get_future() { return pr.get_future(); }
};

// Workers pop from their own deque's back and steal from others' fronts.
class thread_pool {
    struct WorkerQueue {
        std::deque<std::unique_ptr<Job>> q;
        std::mutex m;
    };

    template <class F>
    struct FunctionJob : Job {
        F fn;
        explicit FunctionJob(F f) : fn(std::move(f)) {}
        void execute() override { fn(); }
    };

    std::vector<std::unique_ptr<WorkerQueue>> queues_;
    std::vector<std::thread> workers_;
    const bool stealing_;

    std::mutex wait_m_;
    std::condition_variable cv_, idle_cv_;
    std::atomic<size_t> pending_{0};  // Queued jobs, excluding running jobs.
    std::atomic<size_t> next_worker_{0};
    std::atomic<size_t> unfinished_{0}; // Queued + running jobs.
    bool stop_ = false;              // Protected by wait_m_.

    inline static thread_local thread_pool* tls_pool_ = nullptr;
    inline static thread_local int tls_index_ = -1;

    bool take_job(size_t id, std::unique_ptr<Job>& job) {
        auto& w = *queues_[id];
        std::lock_guard lk(w.m);
        if (w.q.empty()) return false;
        job = std::move(w.q.back());
        w.q.pop_back();
        --pending_;
        return true;
    }

    bool steal_job(int id, std::unique_ptr<Job>& job) {
        const int n = static_cast<int>(queues_.size());
        // id == -1 means an external helper: scan queues 0 through n - 1.
        for (int j = 1; j <= n; ++j) {
            const int victim = (id + j) % n;
            if (victim == id) continue;
            auto& w = *queues_[victim];
            std::unique_lock lk(w.m, std::try_to_lock);
            if (!lk.owns_lock() || w.q.empty()) continue;
            job = std::move(w.q.front());
            w.q.pop_front();
            --pending_;
            return true;
        }
        return false;
    }

    bool work_available(size_t id) {
        if (stealing_) return pending_.load() > 0;
        std::lock_guard lk(queues_[id]->m);
        return !queues_[id]->q.empty();
    }

    void run(std::unique_ptr<Job> job) {
        try {
            job->execute();
        } catch (...) {
            // Keep the worker alive even if a plain Job throws.
        }
        job.reset();
        if (--unfinished_ == 0) {
            std::lock_guard lk(wait_m_);
            idle_cv_.notify_all();
        }
    }

    void worker_loop(unsigned id) {
        tls_pool_ = this;
        tls_index_ = static_cast<int>(id);
        for (;;) {
            std::unique_ptr<Job> job;
            if (take_job(id, job) || (stealing_ && steal_job(id, job))) {
                run(std::move(job));
                continue;
            }
            std::unique_lock lk(wait_m_);
            cv_.wait(lk, [&] { return stop_ || work_available(id); });
            if (stop_ && !work_available(id)) break;
        }
        tls_pool_ = nullptr;
        tls_index_ = -1;
    }

public:
    explicit thread_pool(unsigned n = std::thread::hardware_concurrency(),
                         bool stealing = true)
        : stealing_(stealing) {
        n = n ? n : 1;
        queues_.reserve(n);
        workers_.reserve(n);
        for (unsigned i = 0; i < n; ++i)
            queues_.push_back(std::make_unique<WorkerQueue>());
        try {
            for (unsigned i = 0; i < n; ++i)
                workers_.emplace_back([this, i] { worker_loop(i); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~thread_pool() { shutdown(); }
    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    void push(std::unique_ptr<Job> job) {
        if (!job) throw std::invalid_argument("thread_pool: null job");
        const bool local = tls_pool_ == this && tls_index_ >= 0;
        const size_t id = local ? static_cast<size_t>(tls_index_)
                               : next_worker_.fetch_add(1) % queues_.size();
        {
            // Use the same mutex as the sleep predicate and shutdown.
            std::lock_guard lk(wait_m_);
            if (stop_) throw std::runtime_error("thread_pool: submit after shutdown");
            auto& w = *queues_[id];
            std::lock_guard qlk(w.m);
            w.q.push_back(std::move(job));
            ++pending_;
            ++unfinished_;
        }
        if (stealing_) cv_.notify_one();
        else if (!local) cv_.notify_all(); // Wake the owner of external work.
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args) {
        auto fn = [f = std::forward<F>(f), ... args = std::forward<Args>(args)]()
                  mutable -> decltype(auto) {
            return std::invoke(std::move(f), std::move(args)...);
        };
        using result_type = decltype(fn());
        std::packaged_task<result_type()> task(std::move(fn));
        auto future = task.get_future();
        submit_detached(std::move(task));
        return future;
    }

    template <class F>
    void submit_detached(F&& f) {
        push(std::make_unique<FunctionJob<std::decay_t<F>>>(std::forward<F>(f)));
    }

    // Call from outside a pool task; the running task itself counts as unfinished.
    void wait_idle() {
        std::unique_lock lk(wait_m_);
        idle_cv_.wait(lk, [&] { return unfinished_.load() == 0; });
    }

    // Run tasks while waiting so nested parallel work cannot deadlock the pool.
    template <class Pred>
    void help_while(Pred pred) {
        const int me = tls_pool_ == this ? tls_index_ : -1;
        while (!pred()) {
            std::unique_ptr<Job> job;
            if ((me >= 0 && take_job(me, job)) || steal_job(me, job))
                run(std::move(job));
            else std::this_thread::yield();
        }
    }

    // Call from outside the pool; shutdown must not race another shutdown call.
    void shutdown() {
        {
            std::lock_guard lk(wait_m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_)
            if (t.joinable()) t.join();
    }

    unsigned size() const { return static_cast<unsigned>(queues_.size()); }
    static thread_pool* current() { return tls_pool_; }
    static int current_worker() { return tls_pool_ ? tls_index_ : -1; }
};

}  // namespace tp
