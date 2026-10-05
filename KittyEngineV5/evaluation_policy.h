#pragma once
#include <cstdint>
#include <concepts>
#include <cmath>

namespace bb::evaluation {
  inline constexpr int32_t kCheckmateScore = -100'000;
  inline constexpr int32_t kStalemateScore = 0;

  template <typename EvalPolicy>
  concept EvaluationPolicy = requires(EvalPolicy policy, const BoardState& state) {
    { policy.evaluate(state) } -> std::same_as<int32_t>;
  };
}
