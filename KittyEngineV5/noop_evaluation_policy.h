#pragma once
#include "evaluation_policy.h"

namespace bb::evaluation {
  class NoopEvaluationPolicy {

  public:
    int32_t evaluate(const BoardState&) { return 0;}
    void reset() {}
    template <bool>
    void markPiece(Side, Piece, Square) noexcept {}
  };

  static_assert(EvaluationPolicy<NoopEvaluationPolicy>);
}
