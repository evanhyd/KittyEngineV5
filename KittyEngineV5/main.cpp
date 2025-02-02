#include <iostream>
#include "bitboard.h"
#include "board.h"

using namespace std;

int main() {
  Board board = Board::fromFEN("rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2");
  cout << board << '\n';
}
