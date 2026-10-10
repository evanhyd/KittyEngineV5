#include "perft_driver.h"
#include "position_fens.h"
#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "mlp_evaluation_policy.h"
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
  constexpr float timePercentage = 0.05f;

  try {
#if KITTY_ENABLE_SYZYGY
    tablebase::Service tablebases;
#endif
    Board board{
    searching::NegamaxSearchPolicy{
      evaluation::MLPEvaluationPolicy{"weights.bin"},
      // evaluation::HandCraftEvaluationPolicy{},
      aspirationWindow,
      ttTableSize
#if KITTY_ENABLE_SYZYGY
      , &tablebases
#endif
    },
    time_control::EqualPercentageTimeControlPolicy{
      timePercentage
    }
    };

    user_interface::TerminalUI terminal{ board, cin, cout, cerr };
    terminal.run();

  } catch (const exception& error) {
    std::cerr << error.what() << '\n';
  }
}
