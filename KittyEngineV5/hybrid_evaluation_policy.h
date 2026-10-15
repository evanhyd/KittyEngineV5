#pragma once
#include "evaluation_policy.h"
#include "handcraft_evaluation_policy.h"
#include "nnue_evaluation_policy.h"

namespace bb::evaluation {
  class HybridEvaluationPolicy {
    HandCraftEvaluationPolicy hcEval_;
    NNUEEvaluationPolicy nnueEval_;

  public:
    explicit HybridEvaluationPolicy(const std::filesystem::path& filename)
      : hcEval_{}, nnueEval_{ filename } {
    }

    int32_t evaluate(const BoardState& state) {
      return 0;
    }

    void reset() {
      hcEval_.reset();
      nnueEval_.reset();
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      hcEval_.markPiece<Add>(side, piece, square);
      nnueEval_.markPiece<Add>(side, piece, square);
    }
  };

  static_assert(EvaluationPolicy<HybridEvaluationPolicy>);
}
