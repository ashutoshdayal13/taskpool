#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "thread_pool.hpp"

namespace tp {

// Tasks run after their dependencies finish. Task functions must not throw.
class task_graph {
public:
    explicit task_graph(thread_pool& pool) : pool_(pool) {}

    int add(std::function<void()> f, std::vector<int> deps = {}) {
        const int id = static_cast<int>(nodes_.size());
        auto entry = std::make_unique<node>();
        entry->fn = std::move(f);
        entry->waiting = static_cast<int>(deps.size());
        for (int dependency : deps)
            nodes_[static_cast<size_t>(dependency)]->successors.push_back(id);
        nodes_.push_back(std::move(entry));
        return id;
    }

    // Call once, after adding all nodes.
    void run_and_wait() {
        remaining_ = static_cast<long>(nodes_.size());
        // Collect sources before submitting; running tasks can release successors
        // during the scan and cause them to be submitted twice.
        std::vector<int> sources;
        for (int i = 0; i < static_cast<int>(nodes_.size()); ++i)
            if (nodes_[static_cast<size_t>(i)]->waiting == 0) sources.push_back(i);
        for (int i : sources) pool_.submit_detached([this, i] { exec(i); });
        pool_.help_while([&] { return remaining_.load() == 0; });
    }

private:
    struct node {
        std::function<void()> fn;
        std::atomic<int> waiting{0};
        std::vector<int> successors;
    };
    thread_pool& pool_;
    std::vector<std::unique_ptr<node>> nodes_;
    std::atomic<long> remaining_{0};

    void exec(int i) {
        node& entry = *nodes_[static_cast<size_t>(i)];
        entry.fn();
        for (int successor : entry.successors)
            if (--nodes_[static_cast<size_t>(successor)]->waiting == 0)
                pool_.submit_detached([this, successor] { exec(successor); });
        --remaining_;
    }
};

}  // namespace tp
