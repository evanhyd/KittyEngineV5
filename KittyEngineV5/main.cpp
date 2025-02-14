#include "bitboard.h"
#include "board.h"
#include "perft_driver.h"
#include <iostream>

using namespace std;

int main() {
  const std::string kInitialPositionFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const std::string kKiwipeteFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
  const std::string kRookEndGameFEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ";
  const std::string kMirrorViewFEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
  const std::string kTalkChessBugFEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
  const std::string kStevenAltFEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

  PerftDriver board{ kInitialPositionFEN };
  cout << board << '\n';
  board.perft<false>(7);
}
