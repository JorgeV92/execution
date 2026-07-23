// tests/beman/execution/exec-parallel-scheduler-replacement.test.cpp -*-C++-*-
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <span>
#include <test/execution.hpp>

#ifdef BEMAN_HAS_MODULES
import beman.execution;
#else
#include <beman/execution.hpp>
#endif

namespace {
namespace replacement = test_std::parallel_scheduler_replacement;

struct replacement_backend final : replacement::parallel_scheduler_backend {
    auto schedule(replacement::receiver_proxy& proxy, ::std::span<::std::byte>) noexcept -> void override {
        ++scheduleCalls;
        proxy.set_value();
    }

    auto schedule_bulk_chunked(::std::size_t                          shape,
                               replacement::bulk_item_receiver_proxy& proxy,
                               ::std::span<::std::byte>) noexcept -> void override {
        if (shape != 0uz) {
            proxy.execute(0uz, shape);
        }
        proxy.set_value();
    }

    auto schedule_bulk_unchunked(::std::size_t                          shape,
                                 replacement::bulk_item_receiver_proxy& proxy,
                                 ::std::span<::std::byte>) noexcept -> void override {
        for (::std::size_t index{}; index != shape; ++index) {
            proxy.execute(index, index + 1uz);
        }
        proxy.set_value();
    }

    ::std::atomic<unsigned> scheduleCalls{};
};
} // namespace

namespace beman::execution::parallel_scheduler_replacement {
auto query_parallel_scheduler_backend() -> ::std::shared_ptr<parallel_scheduler_backend> {
    static const auto backend = ::std::make_shared<::replacement_backend>();
    return backend;
}
} // namespace beman::execution::parallel_scheduler_replacement

namespace {
auto testReplacementBackend() -> void {
    const auto backend = ::std::dynamic_pointer_cast<replacement_backend>(
        test_std::parallel_scheduler_replacement::query_parallel_scheduler_backend());
    ASSERT(static_cast<bool>(backend));

    const auto first  = test_std::get_parallel_scheduler();
    const auto second = test_std::get_parallel_scheduler();
    ASSERT(first == second);

    unsigned value{};
    test_std::sync_wait(test_std::schedule(first) | test_std::then([&] { value = 42u; }));
    ASSERT(value == 42u);
    ASSERT(backend->scheduleCalls.load() == 1u);
}
} // namespace

TEST(exec_parallel_scheduler_replacement) { testReplacementBackend(); }
