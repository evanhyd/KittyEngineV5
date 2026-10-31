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
    using PositionHistory = SmallVec<ZobristHash::Hash, kMaxMovePerGame>;
    using PVLine = SmallVec<Move, kMaxDepthHardCutoff + 1>;

    struct SearchParam {
      int maxDepth;
      PositionHistory& history;
    };

    struct SearchResult {
      int32_t score;
      Move move;
      std::span<const Move> pvLine;
      uint64_t nodes;
      std::chrono::nanoseconds searchingTime;
    };


    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<White>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<Black>(state, param) } -> std::same_as<SearchResult>;
      { policy.reset() } -> std::same_as<void>;
      { policy.resetEvaluation() } -> std::same_as<void>;
      { policy.isCheckmateScore(int32_t{}) } -> std::same_as<bool>;
      { policy.template markPiece<true>(White, Pawn, A1) } -> std::same_as<void>;
      { policy.template markPiece<false>(White, Pawn, A1) } -> std::same_as<void>;
    };
  }
}
