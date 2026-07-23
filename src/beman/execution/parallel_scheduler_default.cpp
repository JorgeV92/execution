// src/beman/execution/parallel_scheduler_default.cpp               -*-C++-*-
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef BEMAN_HAS_MODULES
import beman.execution.detail.get_stop_token;
import beman.execution.detail.inplace_stop_source;
import beman.execution.detail.parallel_scheduler_replacement;
#else
#include <beman/execution/detail/parallel_scheduler_replacement.hpp>
#endif

// ----------------------------------------------------------------------------

namespace {
namespace replacement = ::beman::execution::parallel_scheduler_replacement;

#ifdef BEMAN_EXECUTION_TESTING
::std::atomic<bool>          failNextAllocation{};
::std::atomic<bool>          failNextEnqueue{};
::std::atomic<::std::size_t> inPlaceAllocations{};
::std::atomic<::std::size_t> heapAllocations{};
#endif

auto stopRequested(replacement::receiver_proxy& proxy) noexcept -> bool {
    auto token = proxy.template try_query<::beman::execution::inplace_stop_token>(::beman::execution::get_stop_token);
    return token.has_value() && token->stop_requested();
}

struct task {
    task() = default;

    task(const task&) = delete;
    task(task&&)      = delete;

    virtual ~task() = default;

    auto operator=(const task&) -> task& = delete;
    auto operator=(task&&) -> task&      = delete;

    virtual auto run() noexcept -> void     = 0;
    virtual auto destroy() noexcept -> void = 0;
};

template <typename Type>
auto destroyInPlace(Type* object) noexcept -> void {
    ::std::destroy_at(object);
}

template <typename Type>
auto destroyOnHeap(Type* object) noexcept -> void {
    delete object;
}

template <typename Type, typename... Args>
auto makeOperation(::std::span<::std::byte> storage, Args&&... args) -> Type* {
#ifdef BEMAN_EXECUTION_TESTING
    if (failNextAllocation.exchange(false, ::std::memory_order_relaxed)) {
        throw ::std::bad_alloc{};
    }
#endif

    void*         address = storage.data();
    ::std::size_t space   = storage.size();
    if (address != nullptr && ::std::align(alignof(Type), sizeof(Type), address, space) != nullptr) {
#ifdef BEMAN_EXECUTION_TESTING
        inPlaceAllocations.fetch_add(1uz, ::std::memory_order_relaxed);
#endif
        return ::new (address) Type(&destroyInPlace<Type>, ::std::forward<Args>(args)...);
    }

#ifdef BEMAN_EXECUTION_TESTING
    heapAllocations.fetch_add(1uz, ::std::memory_order_relaxed);
#endif
    return new Type(&destroyOnHeap<Type>, ::std::forward<Args>(args)...);
}

struct schedule_task final : task {
    using destroy_fn = void (*)(schedule_task*) noexcept;

    schedule_task(destroy_fn destroy, replacement::receiver_proxy& proxy) noexcept
        : m_destroy(destroy), m_proxy(&proxy) {}

    auto run() noexcept -> void override {
        auto* const proxy   = m_proxy;
        const bool  stopped = stopRequested(*proxy);
        auto* const self    = this;
        m_destroy(self);
        if (stopped) {
            proxy->set_stopped();
        } else {
            proxy->set_value();
        }
    }

    auto destroy() noexcept -> void override { m_destroy(this); }

  private:
    destroy_fn                   m_destroy;
    replacement::receiver_proxy* m_proxy;
};

struct bulk_state {
    using destroy_fn = void (*)(bulk_state*) noexcept;

    bulk_state(destroy_fn destroy, replacement::bulk_item_receiver_proxy& proxy) noexcept
        : m_destroy(destroy), m_proxy(&proxy) {}

    bulk_state(const bulk_state&) = delete;
    bulk_state(bulk_state&&)      = delete;

    ~bulk_state() = default;

    auto operator=(const bulk_state&) -> bulk_state& = delete;
    auto operator=(bulk_state&&) -> bulk_state&      = delete;

    auto addWork() noexcept -> void {
        const ::std::lock_guard guard(m_mutex);
        ++m_outstanding;
    }

    auto recordSubmissionError(const ::std::exception_ptr& error, bool removeWork) noexcept -> void {
        const ::std::lock_guard guard(m_mutex);
        if (!m_error) {
            m_error = error;
        }
        if (removeWork) {
            --m_outstanding;
        }
    }

    auto finishSubmission() noexcept -> void { finish(false, false); }

    auto finishWork(bool stopped) noexcept -> void { finish(stopped, true); }

    auto stopRequested() const noexcept -> bool { return ::stopRequested(*m_proxy); }

    auto execute(::std::size_t begin, ::std::size_t end) noexcept -> void { m_proxy->execute(begin, end); }

  private:
    auto finish(bool stopped, bool workFinished) noexcept -> void {
        replacement::bulk_item_receiver_proxy* proxy{};
        ::std::exception_ptr                   error;
        bool                                   wasStopped{};
        destroy_fn                             destroy{};
        {
            const ::std::lock_guard guard(m_mutex);
            m_stopped = m_stopped || stopped;
            if (workFinished) {
                --m_outstanding;
            } else {
                m_submitting = false;
            }
            if (m_submitting || m_outstanding != 0uz) {
                return;
            }
            proxy      = m_proxy;
            error      = m_error;
            wasStopped = m_stopped;
            destroy    = m_destroy;
        }

        destroy(this);
        if (error) {
            proxy->set_error(error);
        } else if (wasStopped) {
            proxy->set_stopped();
        } else {
            proxy->set_value();
        }
    }

    destroy_fn                             m_destroy;
    replacement::bulk_item_receiver_proxy* m_proxy;
    ::std::mutex                           m_mutex;
    ::std::size_t                          m_outstanding{};
    ::std::exception_ptr                   m_error;
    bool                                   m_stopped{};
    bool                                   m_submitting{true};
};

struct bulk_task final : task {
    bulk_task(bulk_state& state, ::std::size_t begin, ::std::size_t end, bool execute) noexcept
        : m_state(&state), m_begin(begin), m_end(end), m_execute(execute) {}

    auto run() noexcept -> void override {
        auto* const state   = m_state;
        const bool  stopped = state->stopRequested();
        if (!stopped && m_execute) {
            state->execute(m_begin, m_end);
        }
        delete this;
        state->finishWork(stopped);
    }

    auto destroy() noexcept -> void override { delete this; }

  private:
    bulk_state*   m_state;
    ::std::size_t m_begin;
    ::std::size_t m_end;
    bool          m_execute;
};

struct portable_parallel_scheduler_backend final : replacement::parallel_scheduler_backend {
    portable_parallel_scheduler_backend() : portable_parallel_scheduler_backend(defaultWorkerCount()) {}

    explicit portable_parallel_scheduler_backend(::std::size_t workerCount)
        : m_workerCount((::std::max)(workerCount, 1uz)) {
        try {
            m_workers.reserve(m_workerCount);
            for (::std::size_t index{}; index != m_workerCount; ++index) {
                m_workers.emplace_back([this] { runWorker(); });
            }
        } catch (...) {
            requestShutdown();
            joinWorkers();
            throw;
        }
    }

    portable_parallel_scheduler_backend(const portable_parallel_scheduler_backend&) = delete;
    portable_parallel_scheduler_backend(portable_parallel_scheduler_backend&&)      = delete;

    ~portable_parallel_scheduler_backend() override {
        requestShutdown();
        joinWorkers();
    }

    auto operator=(const portable_parallel_scheduler_backend&) -> portable_parallel_scheduler_backend& = delete;
    auto operator=(portable_parallel_scheduler_backend&&) -> portable_parallel_scheduler_backend&      = delete;

    auto schedule(replacement::receiver_proxy& proxy, ::std::span<::std::byte> storage) noexcept -> void override {
        schedule_task* operation{};
        try {
            operation = makeOperation<schedule_task>(storage, proxy);
            enqueue(*operation);
        } catch (...) {
            auto error = ::std::current_exception();
            if (operation != nullptr) {
                operation->destroy();
            }
            proxy.set_error(error);
        }
    }

    auto schedule_bulk_chunked(::std::size_t                          shape,
                               replacement::bulk_item_receiver_proxy& proxy,
                               ::std::span<::std::byte>               storage) noexcept -> void override {
        if (shape == 0uz || stopRequested(proxy)) {
            scheduleBulk(shape, 1uz, proxy, storage, false);
            return;
        }

        const ::std::size_t maxChunks  = m_workerCount > (::std::numeric_limits<::std::size_t>::max)() / 2uz
                                             ? (::std::numeric_limits<::std::size_t>::max)()
                                             : 2uz * m_workerCount;
        const ::std::size_t chunkCount = (::std::min)(shape, maxChunks);
        const ::std::size_t chunkSize  = shape / chunkCount + static_cast<::std::size_t>(shape % chunkCount != 0uz);
        scheduleBulk(shape, chunkSize, proxy, storage, true);
    }

    auto schedule_bulk_unchunked(::std::size_t                          shape,
                                 replacement::bulk_item_receiver_proxy& proxy,
                                 ::std::span<::std::byte>               storage) noexcept -> void override {
        scheduleBulk(shape, 1uz, proxy, storage, shape != 0uz && !stopRequested(proxy));
    }

  private:
    static auto defaultWorkerCount() noexcept -> ::std::size_t {
        const auto count = ::std::thread::hardware_concurrency();
        return count == 0u ? 1uz : static_cast<::std::size_t>(count);
    }

    auto enqueue(task& operation) -> void {
        {
            const ::std::lock_guard guard(m_mutex);
            if (m_shutdownRequested) {
                throw ::std::runtime_error{"parallel scheduler backend is shutting down"};
            }
#ifdef BEMAN_EXECUTION_TESTING
            if (failNextEnqueue.exchange(false, ::std::memory_order_relaxed)) {
                throw ::std::bad_alloc{};
            }
#endif
            m_tasks.push_back(&operation);
        }
        m_condition.notify_one();
    }

    auto scheduleBulk(::std::size_t                          shape,
                      ::std::size_t                          chunkSize,
                      replacement::bulk_item_receiver_proxy& proxy,
                      ::std::span<::std::byte>               storage,
                      bool                                   execute) noexcept -> void {
        bulk_state* state{};
        try {
            state = makeOperation<bulk_state>(storage, proxy);
        } catch (...) {
            proxy.set_error(::std::current_exception());
            return;
        }

        if (!execute) {
            submitBulkTask(*state, 0uz, 0uz, false);
            state->finishSubmission();
            return;
        }

        for (::std::size_t begin{}; begin != shape;) {
            const ::std::size_t end = shape - begin <= chunkSize ? shape : begin + chunkSize;
            if (!submitBulkTask(*state, begin, end, true)) {
                break;
            }
            begin = end;
        }
        state->finishSubmission();
    }

    auto submitBulkTask(bulk_state& state, ::std::size_t begin, ::std::size_t end, bool execute) noexcept -> bool {
        bulk_task* operation{};
        try {
            operation = new bulk_task(state, begin, end, execute);
            state.addWork();
            try {
                enqueue(*operation);
            } catch (...) {
                auto error = ::std::current_exception();
                operation->destroy();
                state.recordSubmissionError(error, true);
                return false;
            }
            return true;
        } catch (...) {
            state.recordSubmissionError(::std::current_exception(), false);
            return false;
        }
    }

    auto requestShutdown() noexcept -> void {
        {
            const ::std::lock_guard guard(m_mutex);
            m_shutdownRequested = true;
        }
        m_condition.notify_all();
    }

    auto joinWorkers() noexcept -> void {
        for (auto& worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    auto runWorker() noexcept -> void {
        while (true) {
            task* operation{};
            {
                ::std::unique_lock guard(m_mutex);
                m_condition.wait(guard, [this] { return m_shutdownRequested || !m_tasks.empty(); });
                if (m_tasks.empty()) {
                    return;
                }
                operation = m_tasks.front();
                m_tasks.pop_front();
            }
            operation->run();
        }
    }

    const ::std::size_t          m_workerCount;
    ::std::mutex                 m_mutex;
    ::std::condition_variable    m_condition;
    ::std::deque<task*>          m_tasks;
    ::std::vector<::std::thread> m_workers;
    bool                         m_shutdownRequested{};
};
} // namespace

namespace beman::execution::parallel_scheduler_replacement {
auto query_parallel_scheduler_backend() -> ::std::shared_ptr<parallel_scheduler_backend> {
    static const auto backend = ::std::make_shared<::portable_parallel_scheduler_backend>();
    return backend;
}
} // namespace beman::execution::parallel_scheduler_replacement

#ifdef BEMAN_EXECUTION_TESTING
namespace beman::execution::detail {
auto makePortableParallelSchedulerBackendForTesting(::std::size_t workerCount)
    -> ::std::shared_ptr<::beman::execution::parallel_scheduler_replacement::parallel_scheduler_backend> {
    return ::std::make_shared<::portable_parallel_scheduler_backend>(workerCount);
}

auto failNextParallelSchedulerAllocationForTesting() noexcept -> void {
    ::failNextAllocation.store(true, ::std::memory_order_relaxed);
}

auto failNextParallelSchedulerEnqueueForTesting() noexcept -> void {
    ::failNextEnqueue.store(true, ::std::memory_order_relaxed);
}

auto resetParallelSchedulerAllocationCountsForTesting() noexcept -> void {
    ::inPlaceAllocations.store(0uz, ::std::memory_order_relaxed);
    ::heapAllocations.store(0uz, ::std::memory_order_relaxed);
}

auto parallelSchedulerInPlaceAllocationsForTesting() noexcept -> ::std::size_t {
    return ::inPlaceAllocations.load(::std::memory_order_relaxed);
}

auto parallelSchedulerHeapAllocationsForTesting() noexcept -> ::std::size_t {
    return ::heapAllocations.load(::std::memory_order_relaxed);
}
} // namespace beman::execution::detail
#endif
