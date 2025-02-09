#include "bitboard.h"
#include "board.h"
#include <iostream>

using namespace std;

int main() {
  Board board = Board::fromFEN(Board::kKiwipeteFEN);
  cout << board << '\n';

  MoveList moveList{};
  board.getPseudoMove<kWhite>(moveList);
  cout << moveList << '\n';
}
