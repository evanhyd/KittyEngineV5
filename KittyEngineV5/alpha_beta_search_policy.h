#pragma once
#include "boardstate.h"

namespace bb::search {

  template <typename Board>
  class SearchEngine {

    template <typename NodeType>
    int32_t search(int depth, int32_t alpha, int32_t beta) {
      Board& board = static_cast<Board&>(*this);


    }
  };
}
