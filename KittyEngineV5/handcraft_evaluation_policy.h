#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include <cstdint>

namespace bb::evaluation {
  namespace internal {
    inline constexpr int32_t kFutilityMovePriority = 0;
    inline constexpr int32_t kKillerMovePriority = 99;
    inline constexpr int32_t kEnpassantPriority = 105;
    inline constexpr int32_t kPromotionPriority = 500;
    inline constexpr int32_t kPrincipalVariationPriority = 10000;
    inline constexpr int32_t kTranspositionPriority = 20000;

    // [attacker][victim] = score
    inline constexpr int32_t kCapturePriorityTable[6][6] = {
      //            Pawn, Knight, Bishop, Rook,  Queen, King
      /* Pawn   */ { 105,   205,    305,   405,   505,    0 },
      /* Knight */ { 104,   204,    304,   404,   504,    0 },
      /* Bishop */ { 103,   203,    303,   403,   503,    0 },
      /* Rook   */ { 102,   202,    302,   402,   502,    0 },
      /* Queen  */ { 101,   201,    301,   401,   501,    0 },
      /* King   */ { 100,   200,    300,   400,   500,    0 }
    };
  }

  class HandCraftEvaluationPolicy {
  public:
    int32_t evaluate(const bb::BoardState& boardState) const noexcept {

      return 0;
    }
  };
}
