#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

#include "thread_pool.hpp"

namespace tp {

// Task DAG on top of the pool.
//
//   tp::task_graph g(pool);
//   int a = g.add(load);
//   int b = g.add(transform, {a});      // runs after a
//   int c = g.add(validate,  {a});
//   g.add(publish, {b, c});             // runs after b AND c
//   g.run_and_wait();
//
// How: each node counts its unfinished predecessors (atomic). When a node
// finishes it decrements each successor's count; whoever hits zero submits
// that successor. No locks at runtime -- edges are fixed before run.
// Nodes must not throw.
class task_graph {
public:
    explicit task_graph(thread_pool& p) : pool_(p) {}

    int add(std::function<void()> f, std::vector<int> deps = {}) {
        int id = static_cast<int>(nodes_.size());
        auto n = std::make_unique<node>();
        n->fn = std::move(f);
        n->waiting = static_cast<int>(deps.size());
        for (int d : deps) nodes_[static_cast<size_t>(d)]->succ.push_back(id);
        nodes_.push_back(std::move(n));
        return id;
    }

    // Call once, after all add()s. Blocks (helping the pool) until done.
    void run_and_wait() {
        remaining_ = static_cast<long>(nodes_.size());
        // Find ALL sources first, THEN submit. If we submitted while
        // scanning, an already-running source could finish, decrement a
        // successor to 0 and submit it -- and our scan would then see that
        // successor's count == 0 and submit it a second time.
        std::vector<int> sources;
        for (int i = 0; i < static_cast<int>(nodes_.size()); ++i)
            if (nodes_[static_cast<size_t>(i)]->waiting == 0) sources.push_back(i);
        for (int i : sources) pool_.submit_detached([this, i] { exec(i); });
        pool_.help_while([&] { return remaining_.load() == 0; });
    }

private:
    struct node {
        std::function<void()> fn;
        std::atomic<int> waiting{0};   // unfinished predecessors
        std::vector<int> succ;         // successors, fixed before run
    };
    thread_pool& pool_;
    std::vector<std::unique_ptr<node>> nodes_;
    std::atomic<long> remaining_{0};

    void exec(int i) {
        node& n = *nodes_[static_cast<size_t>(i)];
        n.fn();
        for (int s : n.succ)
            if (--nodes_[static_cast<size_t>(s)]->waiting == 0)
                pool_.submit_detached([this, s] { exec(s); });
        --remaining_;
    }
};

}  // namespace tp
