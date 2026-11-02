#pragma once
#include <cstdint>

namespace bb::searching {
  struct NegamaxSearchConfig {
    // Reverse Futility Pruning (Static Null Move Pruning)
    int reverseFutilityDepth = 3;
    int32_t reverseFutilityMultiplier = 100;
    int32_t reverseFutilityBase = 0;
    int32_t reverseFutilityImprovingBonus = 30;

    // Null Move Pruning
    int nullMoveDepth = 3;
    int nullMoveBase = 3;
    int nullMoveDepthDivisor = 4;
    int32_t nullMoveEvalDivisor = 200;
    int nullMoveMaxEvalBonus = 2;

    // Futility Pruning
    int futilityDepth = 3;
    int32_t futilityMultiplier = 100;
    int32_t futilityBase = 0;
    int32_t futilityImprovingBonus = 20;

    // SEE Pruning
    int seePruningDepth = 3;
    int32_t seePruningMultiplier = -100;
    int32_t seePruningBase = 0;

    // Late Move Pruning
    int lateMovePruningDepth = 4;
    int lateMovePruningMultiplier = 2;
    int lateMovePruningBase = 6;

    // History Pruning
    int historyPruningDepth = 3;
    int32_t historyPruningMultiplier = -300;
    int32_t historyPruningBase = -500;

    // Late Move Reductions
    int lateMoveReductionDepth = 3;
    double lateMoveReductionBase = 0.7844;
    double lateMoveReductionDivisor = 2.69;
    double lateMoveReductionHistoryK = 1.0 / 1000.0;
    double lateMoveReductionHistoryB = 0.0;

    // Improving Heuristic
    int32_t improvingThreshold = 15;

    // History Heuristics Limits
    int32_t maxQuietHistory = 770;
    int32_t maxRefutationHistory = 1230;
    int32_t maxFollowupHistory = 500;
    int32_t maxCaptureHistory = 400;

    // Safety and Stability Invariants
    static constexpr int kNullMoveMinPieces = 2;
  };
}
