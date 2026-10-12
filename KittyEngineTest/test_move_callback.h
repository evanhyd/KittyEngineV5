#pragma once
#include "bitboard.h"

namespace bb::testing_support {
  struct NoOpMoveCallback {
    template <bool Add>
    constexpr void markPiece(Side, Piece, Square) const noexcept {}

  };
}
