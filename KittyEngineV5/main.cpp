#include "boardstate.h"
#include "perft_driver.h"
#include <iostream>

using namespace std;
using namespace bb;

void runPerft() {
  constexpr perft::Config config{ false, true, false };
  const auto [initialPositionState, initialSide] = BoardState::fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto [kiwipeteState, kiwipeteSide] = BoardState::fromFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");
  const auto [rookEndGameState, rookEndGameSide] = BoardState::fromFEN("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ");

  cout << "Initial Position\n";
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(initialPositionState, initialSide, i);
  }

  cout << "Kiwipete\n";
  for (int i = 1; i <= 6; ++i) {
    perft::runPerft<config>(kiwipeteState, kiwipeteSide, i);
  }

  cout << "Rook Endgame\n";
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(rookEndGameState, rookEndGameSide, i);
  }
}

int main() {
  runPerft();
}
