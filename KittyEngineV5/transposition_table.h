#pragma once
#include "move.h"
#include "zobrist_hash.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <xmmintrin.h>

namespace bb {
  class TranspositionTable {
  public:
    enum class ScoreType : uint8_t {
      None = 0,
      LowerBound = 1,
      Exact = 2,
      UpperBound = 3,
    };

    struct alignas(32) Entry {
      ZobristHash::Hash key = 0;                   // 8 bytes (offset 0..7)
      int32_t depth = 0;                           // 4 bytes (offset 8..9)
      int32_t age = 0;                             // 4 bytes (offset 10..11)
      ScoreType scoreType = ScoreType::None;       // 1 byte  (offset 12)
      int32_t score = 0;                           // 4 bytes (offset 20..23)
      Move bestMove{};                             // 4 bytes (offset 16..19)

      constexpr bool isValid() const noexcept {
        return scoreType != ScoreType::None;
      }
    };

    static_assert(std::is_trivially_copyable_v<Entry>, "entry must be trivially copyable for SIMD");
    static_assert(std::hardware_constructive_interference_size % sizeof(Entry) == 0, "entry must be fit into a cache line");

  private:
    std::unique_ptr<Entry[]> entries_;
    size_t capacity_;
    int16_t currentAge_ = 0;

    [[nodiscard]] constexpr size_t getIndex(ZobristHash::Hash key) const noexcept {
      return key & (capacity_ - 1);
    }

  public:
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

    void prefetch(ZobristHash::Hash key) noexcept {
      _mm_prefetch(reinterpret_cast<const char*>(&entries_[getIndex(key)]), _MM_HINT_T0);
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
  };
}
