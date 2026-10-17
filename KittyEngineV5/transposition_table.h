#pragma once
#include "zobrist_hash.h"
#include "move.h"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace bb {
  class TranspositionTable {
  public:
    enum class ScoreType : uint8_t {
      None = 0,
      LowerBound = 1,
      Exact = 2,
      UpperBound = 3,
    };

    struct Entry {
      ZobristHash::Hash key = 0;                   // 8 bytes (offset 0..7)
      int16_t depth = 0;                           // 2 bytes (offset 8..9)
      int16_t age = 0;                             // 2 bytes (offset 10..11)
      ScoreType scoreType = ScoreType::None;       // 1 byte  (offset 12)
      Move bestMove{};                             // 4 bytes (offset 16..19)
      int32_t score = 0;                           // 4 bytes (offset 20..23)

      constexpr bool isValid() const noexcept {
        return scoreType != ScoreType::None;
      }
    };
    static_assert(sizeof(Entry) == 24, "TranspositionTable::Entry must be 24 bytes");
    static_assert(std::is_trivially_copyable_v<Entry>, "TranspositionTable::Entry must be trivially copyable for SIMD operations");

    explicit TranspositionTable(size_t capacity)
      : entries_(std::make_unique<Entry[]>(capacity)),
        capacity_(capacity) {
      if (!std::has_single_bit(capacity)) {
        throw std::invalid_argument("transposition table entry size must be a power of two");
      }
    }

    void age() noexcept {
      ++currentAge_;
    }

    void put(const Entry& entry) noexcept {
      auto& slot = entries_[getIndex(entry.key)];

      // Empty slot, always write.
      if (!slot.isValid()) {
        slot = entry;
        slot.age = currentAge_;
        return;
      }

      // Same position, take the younger and deeper one.
      if (slot.key == entry.key) {
        const Move bestMove = entry.bestMove != Move{} ? entry.bestMove : slot.bestMove;

        if (entry.depth >= slot.depth ||
            (entry.scoreType == ScoreType::Exact && slot.scoreType != ScoreType::Exact)) {
          slot = entry;
          slot.bestMove = bestMove;
        } else if (slot.bestMove == Move{} && bestMove != Move{}) {
          slot.bestMove = bestMove;
        }
        slot.age = currentAge_;
        return;
      }

      // Different position, take the younger and deeper one.
      const int ageDiff = std::min(currentAge_ - slot.age, 100);
      const int effectiveOldDepth = slot.depth - 2 * ageDiff;
      if (entry.depth >= effectiveOldDepth) {
        slot = entry;
        slot.age = currentAge_;
      }
    }

    const Entry* get(ZobristHash::Hash key) const noexcept {
      const auto& entry = entries_[getIndex(key)];
      if (!entry.isValid() || entry.key != key) {
        return nullptr;
      }
      return &entry;
    }

    void clear() noexcept {
      currentAge_ = 0;
      std::fill_n(entries_.get(), capacity_, Entry{});
    }

  private:
    [[nodiscard]] constexpr size_t getIndex(ZobristHash::Hash key) const noexcept {
      return key & (capacity_ - 1);
    }

    std::unique_ptr<Entry[]> entries_;
    size_t capacity_;
    int16_t currentAge_ = 0;
  };
}
