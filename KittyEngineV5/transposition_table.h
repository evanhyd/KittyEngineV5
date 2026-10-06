#pragma once
#include "zobrist_hash.h"
#include  "move.h"
#include <optional>
#include <memory>

namespace bb {
  class TranspositionTable {
  public:
    enum class ScoreType {
      LowerBound,
      Exact,
      UpperBound,
    };

    struct Entry {
      ZobristHash::Hash key;
      int depth; // number of moves look ahead rooted at this node.
      int32_t score;
      ScoreType scoreType;
      Move bestMove;
    };

    explicit TranspositionTable(size_t capacity)
      : entries_(std::make_unique<std::optional<Entry>[]>(capacity)), capacity_(capacity) {
      if ((capacity & (capacity - 1)) != 0) {
        throw std::invalid_argument("transposition table entry size must be two's power");
      }
    }

    void put(const Entry& entry) noexcept {
      entries_[entry.key & (capacity_-1)] = entry;
    }

    std::optional<Entry> get(ZobristHash::Hash key) const noexcept {
      auto& entry = entries_[key & (capacity_ - 1)];
      if (!entry || entry->key != key) {
        return std::nullopt;
      }
      return entry;
    }

  private:
    std::unique_ptr<std::optional<Entry>[]> entries_;
    size_t capacity_;
  };
}
