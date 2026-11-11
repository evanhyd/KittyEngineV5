#pragma once
#include "evaluation_policy.h"
#include "searching_policy.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace bb::searching {

  class AspirationWindow {
   public:
    AspirationWindow(int depth, int32_t score, int32_t aspirationWindow, int32_t divisor) noexcept
      : window_(aspirationWindow), divisor_(divisor) {
      if (depth <= 4) {
        alpha_ = evaluation::kCheckmateLossScore;
        beta_  = evaluation::kCheckmateWinScore;
        return;
      }

      const int32_t delta = std::max(aspirationWindow, std::abs(score) / divisor_);
      if (score >= 30'000) {
        alpha_ = score - delta;
        beta_  = evaluation::kCheckmateWinScore;
      } else if (score <= -30'000) {
        alpha_ = evaluation::kCheckmateLossScore;
        beta_  = score + delta;
      } else {
        alpha_ = score - delta;
        beta_  = score + delta;
      }
    }

    [[nodiscard]] int32_t alpha() const noexcept { return alpha_; }
    [[nodiscard]] int32_t beta() const noexcept { return beta_; }

    bool expand(int32_t score) noexcept {
      if (isCheckmate(score) || (score > alpha_ && score < beta_)) {
        return false;
      }

      ++attempt_;
      if (attempt_ >= 3) {
        if (score >= beta_) beta_ = evaluation::kCheckmateWinScore;
        if (score <= alpha_) alpha_ = evaluation::kCheckmateLossScore;
        return true;
      }

      const int32_t delta = (attempt_ == 1)
        ? std::max(window_ * 2, (std::abs(score) * 2) / divisor_)
        : std::max(window_ * 4, (std::abs(score) * 4) / divisor_);

      if (score >= beta_) {
        beta_ = (score >= 30'000)
          ? evaluation::kCheckmateWinScore
          : std::min(evaluation::kCheckmateWinScore, beta_ + delta);
      } else if (score <= alpha_) {
        alpha_ = (score <= -30'000)
          ? evaluation::kCheckmateLossScore
          : std::max(evaluation::kCheckmateLossScore, alpha_ - delta);
      }
      return true;
    }

   private:
    int32_t alpha_ = evaluation::kCheckmateLossScore;
    int32_t beta_  = evaluation::kCheckmateWinScore;
    const int32_t window_;
    const int32_t divisor_;
    int attempt_ = 0;

    static bool isCheckmate(int32_t score) noexcept {
      const int32_t absScore = std::abs(score);
      return evaluation::kCheckmateWinScore - 2 * kMaxDepthHardCutoff <= absScore && absScore <= evaluation::kCheckmateWinScore;
    }
  };

}  // namespace bb::searching
