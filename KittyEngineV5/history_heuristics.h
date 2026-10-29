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
    static constexpr int32_t kMaxQuietHistory = 1000;
    static constexpr int32_t kMaxRefutationHistory = 1000;
    static constexpr int32_t kMaxFollowupHistory = 500;
    static constexpr int32_t kMaxCaptureHistory = 400;
    static constexpr int32_t kMaxQuietScore = kMaxQuietHistory + kMaxRefutationHistory + kMaxFollowupHistory;

    using PieceToHistory = std::array<std::array<int16_t, kSquareSize>, kPieceSize>;
    using ContinuationHistory = std::array<std::array<PieceToHistory, kSquareSize>, kPieceSize>;

    std::array<std::array<Move, kMaxKillers>, kMaxDepthHardCutoff + 1> killerMoves_{};
    std::array<std::array<std::array<int16_t, kSquareSize>, kSquareSize>, 2> quietHistory_{};
    std::array<ContinuationHistory, 2> refutation_{}; // 1-ply continuation history by Side
    std::array<ContinuationHistory, 2> followup_{};   // 2-ply continuation history by Side
    std::array<std::array<std::array<std::array<int16_t, kPieceSize>, kSquareSize>, kPieceSize>, 2> captureHistory_{};

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
      applyOffset(quietHistory_[side][move.from()][move.to()], offset, kMaxQuietHistory);
      const Piece piece = move.getMovedPiece();
      const Square to = move.to();
      if (prev1) {
        applyOffset(refutation_[side][prev1.getMovedPiece()][prev1.to()][piece][to], offset, kMaxRefutationHistory);
      }
      if (prev2) {
        applyOffset(followup_[side][prev2.getMovedPiece()][prev2.to()][piece][to], offset, kMaxFollowupHistory);
      }
    }

  public:
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
    void recordCaptureCutoff(Piece attacker, Square to, Piece victim, int depth) noexcept {
      const int32_t reward = std::min(kMaxCaptureHistory, depth * depth);
      applyOffset(captureHistory_[side][attacker][to][victim], reward, kMaxCaptureHistory);
    }

    template <Side side>
    int16_t getCaptureScore(Piece attacker, Square to, Piece victim) const noexcept {
      return captureHistory_[side][attacker][to][victim];
    }

    Move getKiller(int ply, size_t index) const noexcept {
      return killerMoves_[ply][index];
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
      return std::clamp(score, -kMaxQuietScore, kMaxQuietScore);
    }
  };
}
