#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "nnue_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "terminal_ui.h"
#include <algorithm>
#include <bit>
#include <cctype>
#include <iostream>
#include <optional>
#include <span>
#include <string_view>

constexpr bool iequals(std::string_view a, std::string_view b) noexcept {
  return std::ranges::equal(a, b, [](char c1, char c2) {
    return std::tolower(static_cast<unsigned char>(c1)) == std::tolower(static_cast<unsigned char>(c2));
  });
}

// Computes the power-of-2 number of entries whose memory footprint
// (entries * 24 bytes) is closest to targetMb.
consteval size_t ttEntriesForMb(size_t targetMb) noexcept {
  const size_t targetEntries = (targetMb * 1024ULL * 1024ULL) / sizeof(bb::TranspositionTable::Entry);
  const size_t lower = std::bit_floor(targetEntries);
  const size_t higher = lower << 1;
  return (higher - targetEntries < targetEntries - lower) ? higher : lower;
}

constexpr std::optional<size_t> getTTSize(std::string_view mode) noexcept {
  if (iequals(mode, "bullet"))    return ttEntriesForMb(32);   //  32 MB -> 1ULL << 20 entries (~24 MB)
  if (iequals(mode, "blitz"))     return ttEntriesForMb(256);  // 256 MB -> 1ULL << 23 entries (~192 MB)
  if (iequals(mode, "rapid"))     return ttEntriesForMb(1024); //  1 GB  -> 1ULL << 25 entries (~768 MB)
  if (iequals(mode, "classical")) return ttEntriesForMb(2048); //  2 GB  -> 1ULL << 26 entries (~1.5 GB)
  return std::nullopt;
}

void printHelp() {
  std::cout << "Usage: KittyEngine [options]\n\n"
            << "Options:\n"
            << "  --type <hce|nnue>                     Evaluation policy (default: nnue)\n"
            << "  --mode <bullet|blitz|rapid|classical> TT size: bullet=32MB, blitz=256MB, rapid=1GB, classical=2GB (default: rapid)\n"
            << "  --help, -h                            Print CLI help and exit\n";
}

template <typename EvalPolicy>
void runEngine(EvalPolicy&& evalPolicy, size_t ttCapacity) {
  constexpr int aspirationWindow = 80;
  constexpr float timePercentage = 0.05f;

  bb::Board board{
    bb::searching::NegamaxSearchPolicy{
      std::forward<EvalPolicy>(evalPolicy),
      aspirationWindow,
      ttCapacity
    },
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
    if (iequals(arg, "--help") || iequals(arg, "-h")) {
      printHelp();
      return 0;
    }
    if (iequals(arg, "--type") && i + 1 < args.size()) {
      type = args[++i];
    } else if (iequals(arg, "--mode") && i + 1 < args.size()) {
      mode = args[++i];
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << '\n';
      printHelp();
      return 1;
    }
  }

  const auto ttCapacity = getTTSize(mode);
  if (!ttCapacity) {
    std::cerr << "Invalid --mode: " << mode << " (must be 'bullet', 'blitz', 'rapid', or 'classical')\n";
    return 1;
  }

  try {
    if (iequals(type, "nnue")) {
      runEngine(bb::evaluation::NNUEEvaluationPolicy{"weights.bin"}, *ttCapacity);
    } else if (iequals(type, "hce")) {
      runEngine(bb::evaluation::HandCraftEvaluationPolicy{}, *ttCapacity);
    } else {
      std::cerr << "Invalid --type: " << type << " (must be 'hce' or 'nnue')\n";
      return 1;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  return 0;
}
