#pragma once
#include "zobrist_hash.h"
#include "move.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace bb {
  class TranspositionTable {
  public:
    enum class ScoreType : uint8_t {
      LowerBound,
      Exact,
      UpperBound,
    };

    struct Entry {
      ZobristHash::Hash key;
      int depth; // number of moves look ahead rooted at this node.
      int32_t score;
      ScoreType scoreType;
      bool isValid = false;
      Move bestMove;
    };

    explicit TranspositionTable(size_t capacity)
      : entries_(std::make_unique<Entry[]>(capacity)), capacity_(capacity) {
      if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument("transposition table entry size must be two's power");
      }
    }

    void put(const Entry& entry) noexcept {
      auto& slot = entries_[entry.key & (capacity_ - 1)];
      if (!slot.isValid ||
          slot.key != entry.key ||
          entry.depth > slot.depth ||
          (entry.depth == slot.depth && entry.scoreType == ScoreType::Exact && slot.scoreType != ScoreType::Exact)) {
        slot = entry;
        slot.isValid = true;
      }
    }

    const Entry* get(ZobristHash::Hash key) const noexcept {
      const auto& entry = entries_[key & (capacity_ - 1)];
      if (!entry.isValid || entry.key != key) {
        return nullptr;
      }
      return &entry;
    }

    void clear() noexcept {
      for (size_t i = 0; i < capacity_; ++i) {
        entries_[i].isValid = false;
      }
    }

  private:
    std::unique_ptr<Entry[]> entries_;
    size_t capacity_;
  };
}
