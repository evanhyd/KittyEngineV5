#pragma once
#include "boardstate.h"
#include "searching_policy.h"
#include <algorithm>

namespace bb::searching::draws {
  // History contains the current position as its last entry. One earlier
  // occurrence inside this search is a cycle; game history needs two.
  inline bool isRepetition(const PositionHistory& history, int halfmove, int ply) noexcept {
    const size_t count = std::min(history.size(), static_cast<size_t>(halfmove) + 1);
    int matches = 0;
    for (size_t distance = 2; distance < count; distance += 2) {
      if (history[history.size() - 1 - distance] == history.back()) {
        if (distance <= static_cast<size_t>(ply) || ++matches == 2) return true;
      }
    }
    return false;
  }

  inline bool hasRepeated(const PositionHistory& history, int halfmove) noexcept {
    const size_t first = history.size() - std::min(history.size(), static_cast<size_t>(halfmove) + 1);
    for (size_t i = first; i < history.size(); ++i)
      for (size_t j = i + 2; j < history.size(); j += 2)
        if (history[i] == history[j]) return true;
    return false;
  }

}
