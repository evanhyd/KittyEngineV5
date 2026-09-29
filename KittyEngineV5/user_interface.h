#pragma once
#include "notation.h"
#include <iosfwd>

namespace bb { class BoardState; }

namespace bb::user_interface {
  class UserInterface {
  public:
    static void printBoardState(std::ostream& out, const BoardState& boardState, Color sideToMove);
  };

  void printBitboard(bb::Bitboard bitboard);
}
