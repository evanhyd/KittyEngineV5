#pragma once
#include "bitboard.h"
#include <concepts>
#include <optional>
#include <chrono>

namespace bb::time_control {
  struct TimeControl {
    std::optional<std::chrono::milliseconds> wtime;
    std::optional<std::chrono::milliseconds> winc;
    std::optional<std::chrono::milliseconds> btime;
    std::optional<std::chrono::milliseconds> binc;
    std::optional<std::chrono::milliseconds> moveTime;
    std::optional<int> movesToGo;
  };

  template <typename Policy>
  concept TimeControlPolicy = requires(Policy p, std::optional<TimeControl> timeControl, Side side) {
    { p.set(timeControl) } -> std::same_as<void>;
    { p.shouldContinue(side) } -> std::same_as<bool>;
  };
};
