#pragma once
#include "move.h"
#include <cstdint>
#include <optional>
#include <chrono>

namespace bb {
  class BoardState;

  namespace searching {
    struct SearchParam {
      int maxDepth;
      int32_t pastEval;
    };

    struct SearchResult {
      int32_t score;
      std::optional<Move> bestMove;
      uint64_t nodesSearched;
      std::chrono::steady_clock::duration searchingTime;
    };

    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<White>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<Black>(state, param) } -> std::same_as<SearchResult>;
    };
  }
}
