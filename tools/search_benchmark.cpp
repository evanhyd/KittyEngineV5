#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "notation.h"
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace bb;
#if KITTY_ENABLE_SYZYGY
  tablebase::Service tables;
  if (argc > 1 && std::string_view(argv[1]) != "-") tables.setOption("SyzygyPath", argv[1]);
  // Machine-readable adapter probes for the independent Python oracle.
  if (argc > 2 && std::string_view(argv[2]) == "probe") {
    for (std::string fen; std::getline(std::cin, fen);) {
      BoardState state(fen);
      const auto wdl = tables.probeWdl(state);
      MoveList legal;
      if (state.getSideToMove() == White) state.generateMoves<White>(legal);
      else state.generateMoves<Black>(legal);
      const auto root = tables.rankRoot(state, legal, false);
      std::cout << (wdl ? static_cast<int>(*wdl) : 9) << ' ';
      std::cout << (root && root->outcome ? *root->outcome : 9);
      if (root) for (auto move : root->bestMoves) std::cout << ' ' << notation::moveToString(move);
      std::cout << '\n';
    }
    return 0;
  }
#endif
  Board board{searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, 80, 1u << 20
#if KITTY_ENABLE_SYZYGY
      , &tables
#endif
    }, time_control::EqualPercentageTimeControlPolicy{0.05f}};
  struct Case { const char* name; const char* fen; int depth; };
  const Case positions[] = {
    {"start", fen::kStartPosition.data(), 5},
    {"kiwipete", fen::kKiwipete.data(), 4},
    {"middlegame", "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 4 9", 5},
    {"rook", "4k3/8/8/8/8/8/8/R3K3 w - - 17 1", 6},
    {"pawn", "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1", 7}
  };
  int repeats = argc > 2 ? std::stoi(argv[2]) : 5;
  std::cout << "position,repeat,nodes,milliseconds,score,bestmove,tbhits\n";
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (const auto& p : positions) {
      board.reset();
      board.setPosition(p.fen);
      uint64_t nodes = 0, hits = 0;
      const auto start = std::chrono::steady_clock::now();
      const auto result = board.search(p.depth, std::nullopt, [&](const auto& r) {
        nodes += r.nodesSearched;
#if KITTY_ENABLE_SYZYGY
        hits += r.tablebaseHits;
#endif
      });
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      std::cout << p.name << ',' << repeat << ',' << nodes << ',' << ms << ',' << result.score << ','
        << (result.bestMove ? notation::moveToString(*result.bestMove) : "0000") << ',' << hits << '\n';
    }
  }
}
