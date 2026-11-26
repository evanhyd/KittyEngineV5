#pragma once
#include "boardstate.h"
#include "move.h"
#include "searching_policy.h"
#include "transposition_table.h"
#include <cstdint>
#include <functional>
#include <optional>

namespace bb::searching {
  struct Eval {
    int32_t score;
    bool isUnreliable;
  };

  struct TTProbeResult {
    enum class Type : uint8_t { None, Staled, Cutoff };
    Type type;
    TranspositionTable::ScoreType scoreType;
    int32_t score;
    int depth;
    Move move;
  };

  struct SearchContext {
    BoardState& state;
    const int maxDepth;
    PositionHistory& history;
    uint64_t nodes;
    uint64_t probes;
    uint64_t hits;
    int plyReached;
    bool aborted;
    const std::function<bool()>* isTimeUpHard;
  };

  struct SearchFrame {
    Move move{};
    Move excludedMove{};
    uint32_t pvLength = 0;
    std::optional<int32_t> staticScore;
    std::optional<bool> inCheck;

    void clear() noexcept {
      move = Move{};
      excludedMove = Move{};
      pvLength = 0;
      staticScore = std::nullopt;
      inCheck = std::nullopt;
    }
  };
}
