#include "perft_driver.h"
#include "position_fens.h"
#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "terminal_ui.h"
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
  int aspirationWindow = 50;
  Board board{searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, aspirationWindow}};
  user_interface::TerminalUI terminal{board, cin, cout, cerr};
  terminal.run();
}
