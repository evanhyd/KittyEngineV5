#pragma once
#include "bitboard.h"
#include <cstdint>
#include <concepts>
#include <cmath>

namespace bb {
  class BoardState;
}

namespace bb::evaluation {
  inline constexpr int32_t kCheckmateLossScore = -100'000;
  inline constexpr int32_t kCheckmateWinScore = -kCheckmateLossScore;
  inline constexpr int32_t kStalemateScore = 0;

  template <typename EvalPolicy>
  concept EvaluationPolicy = requires(EvalPolicy policy, const BoardState& state) {
    { policy.evaluate(state) } -> std::same_as<int32_t>;
    { policy.reset() } -> std::same_as<void>;
    { policy.template markPiece<true>(White, Pawn, A1) } -> std::same_as<void>;
    { policy.template markPiece<false>(White, Pawn, A1) } -> std::same_as<void>;
  };
}
