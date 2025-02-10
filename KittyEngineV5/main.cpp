#include "bitboard.h"
#include "board.h"
#include <iostream>

using namespace std;

int main() {
  Board board{};
  board.setFEN(Board::kStartFEN);
  cout << board << '\n';

  board.tryMakeMove<kWhite>(Move(E2, E4, kPawn, 0, Move::kDoublePushFlag));
  cout << board << '\n';
}
