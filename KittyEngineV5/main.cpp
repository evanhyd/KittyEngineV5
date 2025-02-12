#include "bitboard.h"
#include "board.h"
#include "perft_driver.h"
#include <iostream>

using namespace std;

int main() {
  PerftDriver board{ Board::kKiwipeteFEN };
  cout << board << '\n';
  board.perft<true>(5);
}
