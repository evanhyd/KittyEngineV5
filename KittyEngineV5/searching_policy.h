#pragma once
#include "move.h"
#include "small_vec.h"
#include <cstdint>
#include <optional>
#include <chrono>
#include <span>

namespace bb {
  class BoardState;

  namespace searching {
    inline constexpr int kMaxDepthHardCutoff = 64;
    inline constexpr size_t kMaxMovePerGame = 512;
    using PVLine = SmallVec<Move, kMaxDepthHardCutoff + 1>;
    using PositionHistory = SmallVec<ZobristHash::Hash, kMaxMovePerGame>;
    using MoveHistory = SmallVec<Move, kMaxMovePerGame>;

    struct SearchParam {
      int maxDepth;                       // initial max depth
      int32_t pastEval;
      PositionHistory& positionHistory;   // input and output
      std::optional<MoveList>& rootMoves; // reuse across iterative deepening
    };

    struct SearchResult {
      int32_t score = 0;
      std::optional<Move> bestMove;
      uint64_t nodesSearched = 0;
      std::chrono::nanoseconds searchingTime{};
      std::span<const Move> pvLine{};
    };



    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<White>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<Black>(state, param) } -> std::same_as<SearchResult>;
      { policy.reset() } -> std::same_as<void>;
      { policy.invalidateEvaluation() } -> std::same_as<void>;
      { policy.template markPiece<true>(White, Pawn, A1) } -> std::same_as<void>;
      { policy.template markPiece<false>(White, Pawn, A1) } -> std::same_as<void>;
    };
  }
}
