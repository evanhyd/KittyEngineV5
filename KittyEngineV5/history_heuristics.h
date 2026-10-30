#pragma once
#include "bitboard.h"
#include "move.h"
#include "searching_policy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace bb::searching {
  class HistoryHeuristics {
    static constexpr size_t kMaxKillers = 2;
    static constexpr int32_t kMaxHistoryScore = 4096;

    using PieceToHistory = std::array<std::array<int16_t, kSquareSize>, kPieceSize>;
    using ContinuationHistory = std::array<std::array<PieceToHistory, kSquareSize>, kPieceSize>;

    int32_t maxQuietHistory_;
    int32_t maxRefutationHistory_;
    int32_t maxFollowupHistory_;
    int32_t maxCaptureHistory_;
    int32_t maxQuietScore_;

    std::array<std::array<Move, kMaxKillers>, kMaxDepthHardCutoff + 1> killerMoves_{};
    std::array<std::array<std::array<int16_t, kSquareSize>, kSquareSize>, 2> quietHistory_{};
    std::array<ContinuationHistory, 2> refutation_{}; // 1-ply continuation history by Side
    std::array<ContinuationHistory, 2> followup_{};   // 2-ply continuation history by Side
    std::array<std::array<std::array<std::array<int16_t, kPieceSize>, kSquareSize>, kPieceSize>, 2> captureHistory_{};
    std::array<int8_t, 2 * kMaxHistoryScore + 1> lateMoveHistoryAdjustments_{};

    static void applyOffset(int16_t& entry, int32_t offset, int32_t limit) noexcept {
      const int32_t clampedOffset = std::clamp(offset, -limit + 1, limit - 1);
      entry = static_cast<int16_t>(entry + clampedOffset - std::abs(clampedOffset) * entry / limit);
    }

    void recordKiller(int ply, Move move) noexcept {
      auto& killers = killerMoves_[ply];
      if (killers[0] == move) {
        return;
      }
      killers[1] = killers[0];
      killers[0] = move;
    }

    template <Side side>
    void updateQuietHistories(Move move, int32_t offset, Move prev1, Move prev2) noexcept {
      applyOffset(quietHistory_[side][move.from()][move.to()], offset, maxQuietHistory_);
      const Piece piece = move.getMovedPiece();
      const Square to = move.to();
      if (prev1) {
        applyOffset(refutation_[side][prev1.getMovedPiece()][prev1.to()][piece][to], offset, maxRefutationHistory_);
      }
      if (prev2) {
        applyOffset(followup_[side][prev2.getMovedPiece()][prev2.to()][piece][to], offset, maxFollowupHistory_);
      }
    }

  public:
    explicit HistoryHeuristics(int32_t maxQuietHistory,
                               int32_t maxRefutationHistory,
                               int32_t maxFollowupHistory,
                               int32_t maxCaptureHistory,
                               double lateMoveReductionHistoryK,
                               double lateMoveReductionHistoryB) noexcept
      : maxQuietHistory_(maxQuietHistory),
        maxRefutationHistory_(maxRefutationHistory),
        maxFollowupHistory_(maxFollowupHistory),
        maxCaptureHistory_(maxCaptureHistory),
        maxQuietScore_(maxQuietHistory + maxRefutationHistory + maxFollowupHistory) {
      for (int32_t score = -kMaxHistoryScore; score <= kMaxHistoryScore; ++score) {
        const int historyAdjustment = std::max(0, static_cast<int>(std::round(-lateMoveReductionHistoryK * score + lateMoveReductionHistoryB)));
        lateMoveHistoryAdjustments_[static_cast<size_t>(score + kMaxHistoryScore)] = static_cast<int8_t>(historyAdjustment);
      }
    }

    void clear() noexcept {
      killerMoves_ = {};
      quietHistory_ = {};
      refutation_ = {};
      followup_ = {};
      captureHistory_ = {};
    }

    void clearKillers() noexcept {
      killerMoves_ = {};
    }

    bool isKillerMove(int ply, Move move) const noexcept {
      return killerMoves_[ply][0] == move || killerMoves_[ply][1] == move;
    }

    Move getKiller(int ply, size_t index) const noexcept {
      return killerMoves_[ply][index];
    }

    int getHistoryReduction(int32_t historyScore) const noexcept {
      const int32_t clampedScore = std::clamp(historyScore, -kMaxHistoryScore, kMaxHistoryScore);
      return lateMoveHistoryAdjustments_[static_cast<size_t>(clampedScore + kMaxHistoryScore)];
    }

    template <Side side>
    void recordQuietCutoff(int ply, int depth, const MoveList& moves, size_t index, Move prev1 = {}, Move prev2 = {}) noexcept {
      const Move move = moves[index];
      recordKiller(ply, move);
      const int32_t reward = depth * depth;
      updateQuietHistories<side>(move, reward, prev1, prev2);

      for (size_t i = 0; i < index; ++i) {
        if (!moves[i].isViolentMove()) {
          updateQuietHistories<side>(moves[i], -reward, prev1, prev2);
        }
      }
    }

    template <Side side>
    int32_t getQuietScore(Move move, Move prev1 = {}, Move prev2 = {}) const noexcept {
      int32_t score = quietHistory_[side][move.from()][move.to()];
      const Piece piece = move.getMovedPiece();
      const Square to = move.to();
      if (prev1) {
        score += refutation_[side][prev1.getMovedPiece()][prev1.to()][piece][to];
      }
      if (prev2) {
        score += followup_[side][prev2.getMovedPiece()][prev2.to()][piece][to];
      }
      return std::clamp(score, -maxQuietScore_, maxQuietScore_);
    }

    template <Side side>
    void recordCaptureCutoff(Piece attacker, Square to, Piece victim, int depth) noexcept {
      const int32_t reward = std::min(maxCaptureHistory_, depth * depth);
      applyOffset(captureHistory_[side][attacker][to][victim], reward, maxCaptureHistory_);
    }

    template <Side side>
    int16_t getCaptureScore(Piece attacker, Square to, Piece victim) const noexcept {
      return captureHistory_[side][attacker][to][victim];
    }
  };
}
