#include "perft_driver.h"
#include "position_fens.h"
#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "terminal_ui.h"
#include <iostream>

using namespace std;
using namespace bb;

void quickPerft() {
  constexpr perft::Config config{ false, true, false };

  cout << "Initial Position\n";
  BoardState state(fen::kStartPosition);
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(state, i);
  }

  cout << "Kiwipete\n";
  state.setPosition(fen::kKiwipete);
  for (int i = 1; i <= 6; ++i) {
    perft::runPerft<config>(state, i);
  }

  cout << "Rook Endgame\n";
  state.setPosition(fen::kRookEndgame);
  for (int i = 1; i <= 7; ++i) {
    perft::runPerft<config>(state, i);
  }
}

int main() {
  //quickPerft();
  constexpr int aspirationWindow = 80;
  constexpr size_t ttTableSize = 1 << 25;
  constexpr size_t [[maybe_unused]] kMemoryUsageMB = ttTableSize * sizeof(std::optional<TranspositionTable::Entry>) / 1024 / 1024;
  constexpr float timePercentage = 0.05f;

  Board board{
    searching::NegamaxSearchPolicy{
      evaluation::HandCraftEvaluationPolicy{},
      aspirationWindow,
      ttTableSize},
    time_control::EqualPercentageTimeControlPolicy{
      timePercentage
    }
  };

  user_interface::TerminalUI terminal{board, cin, cout, cerr};
  terminal.run();
}
