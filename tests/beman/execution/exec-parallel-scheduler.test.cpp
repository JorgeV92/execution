// tests/beman/execution/exec-parallel-scheduler.test.cpp            -*-C++-*-
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <concepts>
#include <cstddef>
#include <exception>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <test/execution.hpp>

#ifdef BEMAN_HAS_MODULES
import beman.execution;
import beman.execution.detail.schedule_result_t;
#else
#include <beman/execution.hpp>
#endif

namespace beman::execution::detail {
auto makePortableParallelSchedulerBackendForTesting(::std::size_t)
    -> ::std::shared_ptr<::beman::execution::parallel_scheduler_replacement::parallel_scheduler_backend>;
auto failNextParallelSchedulerAllocationForTesting() noexcept -> void;
auto failNextParallelSchedulerEnqueueForTesting() noexcept -> void;
auto resetParallelSchedulerAllocationCountsForTesting() noexcept -> void;
auto parallelSchedulerInPlaceAllocationsForTesting() noexcept -> ::std::size_t;
auto parallelSchedulerHeapAllocationsForTesting() noexcept -> ::std::size_t;
} // namespace beman::execution::detail

namespace {
namespace replacement = test_std::parallel_scheduler_replacement;

enum class completion_kind { value, error, stopped };

struct completion_state {
    auto complete(completion_kind kind, const ::std::exception_ptr& error = {}) noexcept -> void {
        {
            const ::std::lock_guard guard(m_mutex);
            ++m_count;
            m_kind  = kind;
            m_error = error;
        }
        m_condition.notify_all();
    }

    auto wait() -> void {
        ::std::unique_lock guard(m_mutex);
        m_condition.wait(guard, [this] { return m_count != 0uz; });
    }

    auto count() const -> ::std::size_t {
        const ::std::lock_guard guard(m_mutex);
        return m_count;
    }

    auto kind() const -> completion_kind {
        const ::std::lock_guard guard(m_mutex);
        return m_kind;
    }

    auto hasError() const -> bool {
        const ::std::lock_guard guard(m_mutex);
        return static_cast<bool>(m_error);
    }

  private:
    mutable ::std::mutex      m_mutex;
    ::std::condition_variable m_condition;
    ::std::size_t             m_count{};
    completion_kind           m_kind{completion_kind::value};
    ::std::exception_ptr      m_error;
};

struct proxy : replacement::receiver_proxy {
    explicit proxy(completion_state&                             completion,
                   ::std::optional<test_std::inplace_stop_token> token = ::std::nullopt) noexcept
        : m_completion(&completion), m_token(token) {}

    auto set_value() noexcept -> void override { m_completion->complete(completion_kind::value); }

    auto set_error(::std::exception_ptr error) noexcept -> void override {
        m_completion->complete(completion_kind::error, error);
    }

    auto set_stopped() noexcept -> void override { m_completion->complete(completion_kind::stopped); }

  private:
    auto query_stop_token() const noexcept -> ::std::optional<test_std::inplace_stop_token> override {
        return m_token;
    }

    completion_state*                             m_completion;
    ::std::optional<test_std::inplace_stop_token> m_token;
};

struct bulk_proxy : replacement::bulk_item_receiver_proxy {
    bulk_proxy(::std::size_t                                 shape,
               completion_state&                             completion,
               bool                                          expectUnchunked = false,
               ::std::optional<test_std::inplace_stop_token> token           = ::std::nullopt)
        : m_shape(shape),
          m_completion(&completion),
          m_coverage(shape),
          m_expectUnchunked(expectUnchunked),
          m_token(token) {}

    auto set_value() noexcept -> void override { m_completion->complete(completion_kind::value); }

    auto set_error(::std::exception_ptr error) noexcept -> void override {
        m_completion->complete(completion_kind::error, error);
    }

    auto set_stopped() noexcept -> void override { m_completion->complete(completion_kind::stopped); }

    auto execute(::std::size_t begin, ::std::size_t end) noexcept -> void override {
        const ::std::lock_guard guard(m_mutex);
        ++m_executeCount;
        if (begin >= end || end > m_shape || (m_expectUnchunked && end != begin + 1uz)) {
            m_invalidRange = true;
            return;
        }
        for (auto index = begin; index != end; ++index) {
            ++m_coverage[index];
        }
    }

    auto hasExactCoverage() const -> bool {
        const ::std::lock_guard guard(m_mutex);
        return !m_invalidRange &&
               ::std::all_of(m_coverage.begin(), m_coverage.end(), [](unsigned count) { return count == 1u; });
    }

    auto executeCount() const -> ::std::size_t {
        const ::std::lock_guard guard(m_mutex);
        return m_executeCount;
    }

  private:
    auto query_stop_token() const noexcept -> ::std::optional<test_std::inplace_stop_token> override {
        return m_token;
    }

    const ::std::size_t                           m_shape;
    completion_state*                             m_completion;
    mutable ::std::mutex                          m_mutex;
    ::std::vector<unsigned>                       m_coverage;
    ::std::size_t                                 m_executeCount{};
    bool                                          m_invalidRange{};
    bool                                          m_expectUnchunked;
    ::std::optional<test_std::inplace_stop_token> m_token;
};

struct backend_synopsis : replacement::parallel_scheduler_backend {
    auto schedule(replacement::receiver_proxy&, ::std::span<::std::byte>) noexcept -> void override {}
    auto schedule_bulk_chunked(::std::size_t,
                               replacement::bulk_item_receiver_proxy&,
                               ::std::span<::std::byte>) noexcept -> void override {}
    auto schedule_bulk_unchunked(::std::size_t,
                                 replacement::bulk_item_receiver_proxy&,
                                 ::std::span<::std::byte>) noexcept -> void override {}
};

struct stopped_receiver_env {
    test_std::inplace_stop_token token;

    auto query(const test_std::get_stop_token_t&) const noexcept -> test_std::inplace_stop_token { return token; }
};

struct stopped_receiver {
    using receiver_concept = test_std::receiver_tag;

    ::std::shared_ptr<completion_state> completion;
    test_std::inplace_stop_token        token;

    auto set_value() && noexcept -> void { completion->complete(completion_kind::value); }

    auto set_error(auto&&) && noexcept -> void { completion->complete(completion_kind::error); }

    auto set_stopped() && noexcept -> void { completion->complete(completion_kind::stopped); }

    auto get_env() const noexcept -> stopped_receiver_env { return {token}; }
};

struct blocking_proxy : replacement::receiver_proxy {
    blocking_proxy(completion_state& completion, ::std::latch& started, ::std::latch& release) noexcept
        : m_completion(&completion), m_started(&started), m_release(&release) {}

    auto set_value() noexcept -> void override {
        m_started->count_down();
        m_release->wait();
        m_completion->complete(completion_kind::value);
    }

    auto set_error(::std::exception_ptr error) noexcept -> void override {
        m_completion->complete(completion_kind::error, error);
    }

    auto set_stopped() noexcept -> void override { m_completion->complete(completion_kind::stopped); }

  private:
    completion_state* m_completion;
    ::std::latch*     m_started;
    ::std::latch*     m_release;
};

auto testParallelSchedulerSynopsis() -> void {
    static_assert(!::std::default_initializable<test_std::parallel_scheduler>);
    static_assert(::std::copy_constructible<test_std::parallel_scheduler>);
    static_assert(::std::move_constructible<test_std::parallel_scheduler>);
    static_assert(test_std::scheduler<test_std::parallel_scheduler>);
    static_assert(::std::same_as<decltype(test_std::get_parallel_scheduler()), test_std::parallel_scheduler>);
    static_assert(::std::same_as<test_std::schedule_result_t<test_std::parallel_scheduler>,
                                 test_std::parallel_scheduler::sender>);
    static_assert(test_std::sender<test_std::parallel_scheduler::sender>);
    static_assert(::std::same_as<decltype(test_std::get_completion_signatures<test_std::parallel_scheduler::sender>()),
                                 test_std::completion_signatures<test_std::set_value_t(),
                                                                 test_std::set_error_t(::std::exception_ptr),
                                                                 test_std::set_stopped_t()>>);
    static_assert(
        noexcept(test_std::get_forward_progress_guarantee(::std::declval<const test_std::parallel_scheduler&>())));
    static_assert(::std::is_abstract_v<replacement::receiver_proxy>);
    static_assert(::std::is_abstract_v<replacement::bulk_item_receiver_proxy>);
    static_assert(::std::is_abstract_v<replacement::parallel_scheduler_backend>);
    static_assert(::std::derived_from<backend_synopsis, replacement::parallel_scheduler_backend>);
    static_assert(::std::same_as<decltype(replacement::query_parallel_scheduler_backend()),
                                 ::std::shared_ptr<replacement::parallel_scheduler_backend>>);
}

auto testDefaultSchedulerAndOrdinarySchedule() -> void {
    const auto first  = test_std::get_parallel_scheduler();
    const auto second = test_std::get_parallel_scheduler();
    ASSERT(first == second);

    const auto caller = ::std::this_thread::get_id();
    auto       worker = caller;
    test_std::sync_wait(test_std::schedule(first) | test_std::then([&] { worker = ::std::this_thread::get_id(); }));
    ASSERT(worker != caller);

    ::std::atomic<unsigned> completions{};
    for (unsigned index{}; index != 64u; ++index) {
        test_std::sync_wait(test_std::schedule(first) | test_std::then([&] { ++completions; }));
    }
    ASSERT(completions.load() == 64u);
}

auto testConcurrentSubmissions() -> void {
    constexpr ::std::size_t THREAD_COUNT          = 8uz;
    constexpr ::std::size_t OPERATIONS_PER_THREAD = 16uz;

    const auto                   scheduler = test_std::get_parallel_scheduler();
    ::std::barrier               start(static_cast<::std::ptrdiff_t>(THREAD_COUNT));
    ::std::atomic<::std::size_t> completions{};
    ::std::vector<::std::thread> submitters;
    submitters.reserve(THREAD_COUNT);
    for (::std::size_t threadIndex{}; threadIndex != THREAD_COUNT; ++threadIndex) {
        submitters.emplace_back([&] {
            start.arrive_and_wait();
            for (::std::size_t index{}; index != OPERATIONS_PER_THREAD; ++index) {
                test_std::sync_wait(test_std::schedule(scheduler) | test_std::then([&] { ++completions; }));
            }
        });
    }
    for (auto& submitter : submitters) {
        submitter.join();
    }
    ASSERT(completions.load() == THREAD_COUNT * OPERATIONS_PER_THREAD);
}

auto testFrontendBulk() -> void {
    const auto scheduler = test_std::get_parallel_scheduler();
    for (const auto shape : {0uz, 1uz, 3uz, 257uz}) {
        ::std::vector<unsigned> coverage(shape);
        ::std::mutex            mutex;
        test_std::sync_wait(test_std::schedule(scheduler) |
                            test_std::bulk(test_std::par, shape, [&](::std::size_t index) noexcept {
                                const ::std::lock_guard guard(mutex);
                                ++coverage[index];
                            }));
        ASSERT(::std::all_of(coverage.begin(), coverage.end(), [](unsigned count) { return count == 1u; }));

        ::std::fill(coverage.begin(), coverage.end(), 0u);
        test_std::sync_wait(
            test_std::schedule(scheduler) |
            test_std::bulk_chunked(test_std::par, shape, [&](::std::size_t begin, ::std::size_t end) noexcept {
                const ::std::lock_guard guard(mutex);
                for (auto index = begin; index != end; ++index) {
                    ++coverage[index];
                }
            }));
        ASSERT(::std::all_of(coverage.begin(), coverage.end(), [](unsigned count) { return count == 1u; }));

        ::std::fill(coverage.begin(), coverage.end(), 0u);
        test_std::sync_wait(test_std::schedule(scheduler) |
                            test_std::bulk_unchunked(test_std::par, shape, [&](::std::size_t index) noexcept {
                                const ::std::lock_guard guard(mutex);
                                ++coverage[index];
                            }));
        ASSERT(::std::all_of(coverage.begin(), coverage.end(), [](unsigned count) { return count == 1u; }));
    }

    ::std::vector<unsigned> sequential(17uz);
    test_std::sync_wait(
        test_std::schedule(scheduler) |
        test_std::bulk(test_std::seq, sequential.size(), [&](::std::size_t index) noexcept { ++sequential[index]; }));
    ASSERT(::std::all_of(sequential.begin(), sequential.end(), [](unsigned count) { return count == 1u; }));
}

auto testBackendBulkRanges() -> void {
    auto backend = replacement::query_parallel_scheduler_backend();
    for (const auto shape : {0uz, 1uz, 3uz, 257uz}) {
        {
            completion_state                       completion;
            bulk_proxy                             receiver(shape, completion);
            alignas(::std::max_align_t)::std::byte storage[512uz];
            backend->schedule_bulk_chunked(shape, receiver, storage);
            completion.wait();
            ASSERT(completion.count() == 1uz);
            ASSERT(completion.kind() == completion_kind::value);
            ASSERT(receiver.hasExactCoverage());
            ASSERT(receiver.executeCount() != 0uz || shape == 0uz);
        }
        {
            completion_state                       completion;
            bulk_proxy                             receiver(shape, completion, true);
            alignas(::std::max_align_t)::std::byte storage[512uz];
            backend->schedule_bulk_unchunked(shape, receiver, storage);
            completion.wait();
            ASSERT(completion.count() == 1uz);
            ASSERT(completion.kind() == completion_kind::value);
            ASSERT(receiver.hasExactCoverage());
            ASSERT(receiver.executeCount() == shape);
        }
    }
}

auto testStopsBeforeStart() -> void {
    test_std::inplace_stop_source source;
    ASSERT(source.request_stop());

    auto completion = ::std::make_shared<completion_state>();
    auto operation  = test_std::connect(test_std::schedule(test_std::get_parallel_scheduler()),
                                       stopped_receiver{completion, source.get_token()});
    test_std::start(operation);
    completion->wait();
    ASSERT(completion->count() == 1uz);
    ASSERT(completion->kind() == completion_kind::stopped);
}

auto testStopWhileQueued() -> void {
    auto backend = test_detail::makePortableParallelSchedulerBackendForTesting(1uz);

    completion_state                       blockerCompletion;
    ::std::latch                           blockerStarted(1);
    ::std::latch                           blockerRelease(1);
    blocking_proxy                         blocker(blockerCompletion, blockerStarted, blockerRelease);
    alignas(::std::max_align_t)::std::byte blockerStorage[512uz];
    backend->schedule(blocker, blockerStorage);
    blockerStarted.wait();

    test_std::inplace_stop_source          source;
    completion_state                       stoppedCompletion;
    proxy                                  stopped(stoppedCompletion, source.get_token());
    alignas(::std::max_align_t)::std::byte stoppedStorage[512uz];
    backend->schedule(stopped, stoppedStorage);
    ASSERT(source.request_stop());

    blockerRelease.count_down();
    blockerCompletion.wait();
    stoppedCompletion.wait();
    ASSERT(blockerCompletion.kind() == completion_kind::value);
    ASSERT(stoppedCompletion.count() == 1uz);
    ASSERT(stoppedCompletion.kind() == completion_kind::stopped);
}

auto testScratchStorageAndFailures() -> void {
    auto backend = test_detail::makePortableParallelSchedulerBackendForTesting(1uz);

    test_detail::resetParallelSchedulerAllocationCountsForTesting();
    {
        completion_state                       completion;
        proxy                                  receiver(completion);
        alignas(::std::max_align_t)::std::byte storage[512uz];
        backend->schedule(receiver, storage);
        completion.wait();
        ASSERT(completion.kind() == completion_kind::value);
        ASSERT(completion.count() == 1uz);
    }
    ASSERT(test_detail::parallelSchedulerInPlaceAllocationsForTesting() == 1uz);
    ASSERT(test_detail::parallelSchedulerHeapAllocationsForTesting() == 0uz);

    test_detail::resetParallelSchedulerAllocationCountsForTesting();
    {
        completion_state completion;
        proxy            receiver(completion);
        backend->schedule(receiver, {});
        completion.wait();
        ASSERT(completion.kind() == completion_kind::value);
        ASSERT(completion.count() == 1uz);
    }
    ASSERT(test_detail::parallelSchedulerInPlaceAllocationsForTesting() == 0uz);
    ASSERT(test_detail::parallelSchedulerHeapAllocationsForTesting() == 1uz);

    {
        test_detail::failNextParallelSchedulerAllocationForTesting();
        completion_state                       completion;
        proxy                                  receiver(completion);
        alignas(::std::max_align_t)::std::byte storage[512uz];
        backend->schedule(receiver, storage);
        completion.wait();
        ASSERT(completion.kind() == completion_kind::error);
        ASSERT(completion.hasError());
        ASSERT(completion.count() == 1uz);
    }
    {
        test_detail::failNextParallelSchedulerEnqueueForTesting();
        completion_state                       completion;
        proxy                                  receiver(completion);
        alignas(::std::max_align_t)::std::byte storage[512uz];
        backend->schedule(receiver, storage);
        completion.wait();
        ASSERT(completion.kind() == completion_kind::error);
        ASSERT(completion.hasError());
        ASSERT(completion.count() == 1uz);
    }
    {
        test_detail::failNextParallelSchedulerEnqueueForTesting();
        completion_state                       completion;
        bulk_proxy                             receiver(8uz, completion);
        alignas(::std::max_align_t)::std::byte storage[512uz];
        backend->schedule_bulk_chunked(8uz, receiver, storage);
        completion.wait();
        ASSERT(completion.kind() == completion_kind::error);
        ASSERT(completion.hasError());
        ASSERT(completion.count() == 1uz);
        ASSERT(receiver.executeCount() == 0uz);
    }
}

auto testBackendDestructionDrainsAndJoins() -> void {
    auto backend = test_detail::makePortableParallelSchedulerBackendForTesting(1uz);

    completion_state                       blockerCompletion;
    ::std::latch                           blockerStarted(1);
    ::std::latch                           blockerRelease(1);
    blocking_proxy                         blocker(blockerCompletion, blockerStarted, blockerRelease);
    alignas(::std::max_align_t)::std::byte blockerStorage[512uz];
    backend->schedule(blocker, blockerStorage);
    blockerStarted.wait();

    completion_state                       queuedCompletion;
    proxy                                  queued(queuedCompletion);
    alignas(::std::max_align_t)::std::byte queuedStorage[512uz];
    backend->schedule(queued, queuedStorage);

    ::std::latch  destructionStarted(1);
    ::std::latch  destructionFinished(1);
    ::std::thread destroyer(
        [ownedBackend = ::std::move(backend), &destructionStarted, &destructionFinished]() mutable {
            destructionStarted.count_down();
            ownedBackend.reset();
            destructionFinished.count_down();
        });
    destructionStarted.wait();
    blockerRelease.count_down();
    blockerCompletion.wait();
    queuedCompletion.wait();
    destructionFinished.wait();
    destroyer.join();

    ASSERT(blockerCompletion.kind() == completion_kind::value);
    ASSERT(queuedCompletion.kind() == completion_kind::value);
    ASSERT(blockerCompletion.count() == 1uz);
    ASSERT(queuedCompletion.count() == 1uz);
}
} // namespace

TEST(exec_parallel_scheduler) {
    testParallelSchedulerSynopsis();
    testDefaultSchedulerAndOrdinarySchedule();
    testConcurrentSubmissions();
    testFrontendBulk();
    testBackendBulkRanges();
    testStopsBeforeStart();
    testStopWhileQueued();
    testScratchStorageAndFailures();
    testBackendDestructionDrainsAndJoins();
}
