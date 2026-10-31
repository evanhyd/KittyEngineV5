#pragma once
#include <cstdint>

namespace bb::searching {
  struct NegamaxSearchConfig {
    // Reverse Futility Pruning (Static Null Move Pruning)
    int reverseFutilityDepth = 3;
    int32_t reverseFutilityMarginPerDepth = 100;

    // Null Move Pruning
    int nullMoveDepth = 2;

    // Futility Pruning
    int futilityDepth = 3;
    int32_t futilityMarginPerDepth = 100;

    // SEE Pruning
    int seePruningDepth = 3;
    int32_t seePruningMarginPerDepth = -100;

    // Late Move Pruning
    int lateMovePruningDepth = 4;
    int lateMovePruningBase = 6;
    int lateMovePruningMultiplier = 2;
    int32_t lateMovePruningStaticScoreGainedThreshold = -30;

    // Late Move Reductions
    int lateMoveReductionDepth = 3;
    double lateMoveReductionBase = 0.7844;
    double lateMoveReductionDivisor = 2.69;
    double lateMoveReductionHistoryK = 1.0 / 1000.0;
    double lateMoveReductionHistoryB = 0.0;
    int32_t lateMoveReductionStaticScoreGainedThreshold = -20;

    // History Heuristics Limits
    int32_t maxQuietHistory = 770;
    int32_t maxRefutationHistory = 1230;
    int32_t maxFollowupHistory = 500;
    int32_t maxCaptureHistory = 400;

    // Safety and Stability Invariants
    static constexpr int kNullMoveReductionDivisor = 2;
    static constexpr int kNullMoveMinPieces = 2;
  };
}
