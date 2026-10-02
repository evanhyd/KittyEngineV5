#pragma once
#include "move.h"
#include <cstdint>
#include <optional>

namespace bb {
  class BoardState;

  namespace searching {
    struct SearchParam {
      int maxDepth;
      int32_t historicalEval;
      std::optional<Move> pvMove;
    };

    struct SearchResult {
      int32_t score;
      std::optional<Move> bestMove;
    };

    template <typename SearchPolicy>
    concept SearchingPolicy = requires(SearchPolicy policy, BoardState& state, const SearchParam& param) {
      { policy.search<kWhite>(state, param) } -> std::same_as<SearchResult>;
      { policy.search<kBlack>(state, param) } -> std::same_as<SearchResult>;
    };
  }
}
