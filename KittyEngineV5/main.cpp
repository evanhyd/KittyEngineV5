#include "board.h"
#include "perft_driver.h"
#include <iostream>

using namespace std;

template <PerftDriver::Config config>
void perft() {
  const BoardState initialPositionState = BoardState::fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const BoardState kiwipeteState = BoardState::fromFEN("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ");
  const BoardState rookEndGameState = BoardState::fromFEN("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ");
  const BoardState mirrorViewState = BoardState::fromFEN("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
  const BoardState talkChessBugState = BoardState::fromFEN("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
  const BoardState stevenAltState = BoardState::fromFEN("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10");

  PerftDriver driver{};

  cout << "Initial Position\n";
  for (int i = 1; i <= 7; ++i) {
    driver.perft<config>(initialPositionState, i);
    driver.perft<config>(initialPositionState, i);
  }

  cout << "Kiwipete\n";
  for (int i = 1; i <= 6; ++i) {
    driver.perft<config>(kiwipeteState, i);
    driver.perft<config>(kiwipeteState, i);
  }

  cout << "Rook Endgame\n";
  for (int i = 1; i <= 7; ++i) {
    driver.perft<config>(rookEndGameState, i);
    driver.perft<config>(rookEndGameState, i);
  }
}

int main() {
  perft<PerftDriver::Config{false, true, false} > ();
}
