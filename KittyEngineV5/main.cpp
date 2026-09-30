#include "perft_driver.h"
#include "position_fens.h"
#include "board.h"
#include "alpha_beta_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "terminal_interface_policy.h"
#include <iostream>

using namespace std;
using namespace bb;

void runPerft() {
  static constexpr perft::Config config{ false, true, false };
  const BoardState initialPositionState{fen::kStartPosition};
  const BoardState kiwipeteState{fen::kKiwipete};
  const BoardState rookEndGameState{fen::kRookEndgame};

  cout << "Initial Position\n";
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(initialPositionState, i);
  }

  cout << "Kiwipete\n";
  for (int i = 1; i <= 6; ++i) {
    perft::runPerft<config>(kiwipeteState, i);
  }

  cout << "Rook Endgame\n";
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(rookEndGameState, i);
  }
}

int main() {
  Board board(
    searching::AlphaBetaSearchingPolicy{},
    evaluation::HandCraftEvaluationPolicy{},
    user_interface::TerminalInterfacePolicy{cin, cout, cerr}
  );
  board.run();
}
