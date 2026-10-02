#pragma once
#include <cstdint>
#include <concepts>

namespace bb::evaluation {
  inline constexpr int32_t kCheckmateScore = -32'000; // Keep mate score small to accommodate aspiration window and depth adjustments.
  inline constexpr int32_t kStalemateScore = 0;

  template <typename EvalPolicy>
  concept EvaluationPolicy = requires(EvalPolicy policy, const BoardState& state) {
    { policy.evaluate(state) } -> std::same_as<int32_t>;
  };
}
