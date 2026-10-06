#pragma once
#include "move.h"
#include "small_vec.h"
#include <cstdint>
#include <optional>
#include <chrono>

namespace bb {
  class BoardState;

  namespace searching {
    inline constexpr int kMaxDepthHardCutoff = 64;
    using PVLine = SmallVec<Move, kMaxDepthHardCutoff + 1>;

    struct SearchParam {
      int maxDepth;
      int32_t pastEval;
      const PVLine& pvLine;
    };

    struct SearchResult {
      int32_t score;
      std::optional<Move> bestMove;
      uint64_t nodesSearched;
      std::chrono::nanoseconds searchingTime;
      PVLine pvLine;
    };

    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<White>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<Black>(state, param) } -> std::same_as<SearchResult>;
    };
  }
}
