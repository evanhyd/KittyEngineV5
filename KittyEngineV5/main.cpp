#include "perft_driver.h"
#include "position_fens.h"
#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "terminal_ui.h"
#include <iostream>

using namespace std;
using namespace bb;

int main() {
  int aspirationWindow = 80;
  Board board{searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, aspirationWindow}};
  user_interface::TerminalUI terminal{board, cin, cout, cerr};
  terminal.run();
}
