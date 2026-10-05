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
      : entries_(std::make_unique<std::optional<Entry>[]>(capacity)), capacity_(capacity){
    }

    void put(const Entry& entry) {
      entries_[entry.key % capacity_] = entry;
    }

    std::optional<Entry> get(ZobristHash::Hash key) const {
      auto& entry = entries_[key % capacity_];
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
