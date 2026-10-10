#pragma once
#include "build_config.h"
#include "move.h"
#include "small_vec.h"
#include <cstdint>
#include <optional>
#include <chrono>

namespace bb {
  class BoardState;

  namespace searching {
    inline constexpr int kMaxDepthHardCutoff = 64;
    inline constexpr size_t kMaxMovePerGame = 1024;
    using PVLine = SmallVec<Move, kMaxDepthHardCutoff + 1>;
    using PositionHistory = SmallVec<ZobristHash::Hash, kMaxMovePerGame>;
    using MoveHistory = SmallVec<Move, kMaxMovePerGame>;

    struct SearchParam {
      int maxDepth;                     // initial max depth
      int32_t pastEval;
      PositionHistory& positionHistory; // input and output
      PVLine& pvLine;                   // input and output
    };

    struct SearchResult {
      int32_t score;
      std::optional<Move> bestMove;
      uint64_t nodesSearched;
      std::chrono::nanoseconds searchingTime;
      const PVLine* pvLine;
#if KITTY_ENABLE_SYZYGY
      uint64_t tablebaseHits = 0; // Successful probes in this iteration only.
#endif
    };

    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<White>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<Black>(state, param) } -> std::same_as<SearchResult>;
      { policy.reset() } -> std::same_as<void>;
      { policy.invalidateEvaluation() } -> std::same_as<void>;
      { policy.template markPiece<true>(White, Pawn, A1) } -> std::same_as<void>;
      { policy.template markPiece<false>(White, Pawn, A1) } -> std::same_as<void>;
      { policy.markCastle(CastlePermission{}) } -> std::same_as<void>;
      { policy.markEnpassant(A3) } -> std::same_as<void>;
    };
  }
}
