#include "bitboard.h"
#include "board.h"
#include "perft_driver.h"
#include <iostream>

using namespace std;

template <bool verbose>
void perft() {
  const string kInitialPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const string kKiwipeteFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
  const string kRookEndGameFEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ";
  const string kMirrorViewFEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
  const string kTalkChessBugFEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
  const string kStevenAltFEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

  cout << "start perft\n";

  PerftDriver board{ kInitialPositionFEN };
  for (int i = 0; i < 4; ++i) {
    board.perft<verbose>(6);
  }

  for (int i = 0; i < 2; ++i) {
    board.perft<verbose>(7);
  }

  board.setFEN(kRookEndGameFEN);
  for (int i = 0; i < 2; ++i) {
    board.perft<verbose>(7);
  }

  board.setFEN(kKiwipeteFEN);
  for (int i = 0; i < 2; ++i) {
    board.perft<verbose>(6);
  }
}

int main() {
  perft<false>();
}
