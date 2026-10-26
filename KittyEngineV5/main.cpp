#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "nnue_evaluation_policy.h"
#include "noop_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "terminal_ui.h"
#include <algorithm>
#include <bit>
#include <cctype>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>

consteval size_t ttEntriesForMb(size_t targetMb) noexcept {
  const size_t targetEntries = (targetMb * 1024ULL * 1024ULL) / sizeof(bb::TranspositionTable::Entry);
  const size_t lower = std::bit_floor(targetEntries);
  const size_t higher = lower << 1;
  return (higher - targetEntries < targetEntries - lower) ? higher : lower;
}

constexpr std::optional<size_t> getTTSize(std::string_view mode) noexcept {
  if (mode == "bullet")    return ttEntriesForMb(64);
  if (mode == "blitz")     return ttEntriesForMb(256);
  if (mode == "rapid")     return ttEntriesForMb(1024);
  if (mode == "classical") return ttEntriesForMb(2048);
  return std::nullopt;
}

void printHelp() {
  std::cout << "Usage: KittyEngine [options]\n\n"
            << "Options:\n"
            << "  --type <hce|nnue|noop>                Evaluation policy: hce, nnue(default), noop\n"
            << "  --mode <bullet|blitz|rapid|classical> TT size: bullet=128MB, blitz=256MB, rapid=1GB(default), classical=2GB\n"
            << "  --help, -h                            Print CLI help and exit\n";
}

template <typename EvalPolicy>
void runEngine(EvalPolicy&& evalPolicy, size_t ttCapacity) {
  constexpr int aspirationWindow = 40;
  constexpr float timePercentage = 0.05f;

  bb::Board board{
    bb::searching::NegamaxSearchPolicy{std::forward<EvalPolicy>(evalPolicy), aspirationWindow, ttCapacity},
    bb::time_control::EqualPercentageTimeControlPolicy{timePercentage}
  };

  bb::user_interface::TerminalUI terminal{ board, std::cin, std::cout, std::cerr };
  terminal.run();
}

int main(int argc, char* argv[]) {
  std::string_view type = "nnue";
  std::string_view mode = "rapid";

  const std::span args(argv + 1, static_cast<size_t>(std::max(0, argc - 1)));
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--help" || arg == "-h") {
      printHelp();
      return 0;
    }
    if (arg == "--type" && i + 1 < args.size()) {
      type = args[++i];
    } else if (arg == "--mode" && i + 1 < args.size()) {
      mode = args[++i];
    } else {
      printHelp();
      return 1;
    }
  }

  const auto ttCapacity = getTTSize(mode);
  if (!ttCapacity) {
    printHelp();
    return 1;
  }

  try {
    if (type == "nnue") {
      runEngine(bb::evaluation::NNUEEvaluationPolicy{"weights.bin"}, *ttCapacity);
    } else if (type == "hce") {
      runEngine(bb::evaluation::HandCraftEvaluationPolicy{}, *ttCapacity);
    } else if (type == "noop") {
      runEngine(bb::evaluation::NoopEvaluationPolicy{}, *ttCapacity);
    } else {
      printHelp();
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
// 296000
// 301244
