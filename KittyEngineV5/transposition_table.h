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
      ZobristHash::Hash key = 0;              // 8 bytes (0..7)
      int32_t score = 0;                      // 4 bytes (8..11)
      int32_t staticScore = 0;                // 4 bytes (12..15)
      Move bestMove{};                        // 4 bytes (16..19)
      int16_t depth = 0;                      // 2 bytes (20..21)
      uint16_t age = 0;                       // 2 bytes (22..23)
      ScoreType scoreType = ScoreType::None;  // 1 byte  (24)
      bool hasStaticScore = false;            // 1 byte  (25)
      bool inCheck = false;                   // 1 byte  (26)
      bool hasInCheck = false;                // 1 byte  (27)
      bool isPV = false;                      // 1 byte  (28)
      // 3 bytes padding (29..31)

      constexpr bool isValid() const noexcept {
        return scoreType != ScoreType::None;
      }
    };
    static_assert(sizeof(Entry) == 32, "Entry must be exactly 32 bytes");

    static_assert(std::is_trivially_copyable_v<Entry>, "entry must be trivially copyable for SIMD");
    static_assert(std::hardware_constructive_interference_size % sizeof(Entry) == 0, "entry must be fit into a cache line");

  private:
    std::unique_ptr<Entry[]> entries_;
    const size_t capacity_;
    uint16_t currentAge_ = 0;

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

    template <bool isEntryPV = false, ScoreType entryScoreType = ScoreType::None>
    void put(const Entry& entry) noexcept {
      auto& slot = entries_[getIndex(entry.key)];

      // Empty slot, always write.
      if (!slot.isValid()) {
        slot = entry;
        slot.age = currentAge_;
        return;
      }

      // Same position, prioritize exact/PV evaluations and deeper searches.
      if (slot.key == entry.key) {
        const Move bestMove = entry.bestMove != Move{} ? entry.bestMove : slot.bestMove;
        const bool hasStaticScore = entry.hasStaticScore || slot.hasStaticScore;
        const int32_t staticScore = entry.hasStaticScore ? entry.staticScore : slot.staticScore;
        const bool inCheck = entry.hasInCheck ? entry.inCheck : slot.inCheck;
        const bool hasInCheck = entry.hasInCheck || slot.hasInCheck;
        const bool isPV = isEntryPV || entry.isPV || slot.isPV;

        bool shouldReplace = false;
        if constexpr (entryScoreType == ScoreType::Exact) {
          if (slot.scoreType != ScoreType::Exact) {
            shouldReplace = true;
          }
        } else if constexpr (entryScoreType == ScoreType::None) {
          if (entry.scoreType == ScoreType::Exact && slot.scoreType != ScoreType::Exact) {
            shouldReplace = true;
          }
        }

        if (!shouldReplace) {
          if constexpr (isEntryPV) {
            if (!slot.isPV) {
              shouldReplace = (entry.depth >= slot.depth - 1);
            } else {
              shouldReplace = (entry.depth >= slot.depth);
            }
          } else {
            if (entry.isPV) {
              if (!slot.isPV) {
                shouldReplace = (entry.depth >= slot.depth - 1);
              } else {
                shouldReplace = (entry.depth >= slot.depth);
              }
            } else if (slot.isPV) {
              shouldReplace = (entry.depth >= slot.depth + 4);
            } else {
              shouldReplace = (entry.depth >= slot.depth);
            }
          }
        }

        if (shouldReplace) {
          slot = entry;
          slot.bestMove = bestMove;
        } else if (slot.bestMove == Move{} && bestMove != Move{}) {
          slot.bestMove = bestMove;
        }
        slot.isPV = isPV;
        slot.hasStaticScore = hasStaticScore;
        slot.staticScore = staticScore;
        slot.inCheck = inCheck;
        slot.hasInCheck = hasInCheck;
        slot.age = currentAge_;
        return;
      }

      // Different position (Hash Collision): protect fresh PV nodes from eviction.
      const int ageDiff = std::clamp(static_cast<int>(currentAge_) - static_cast<int>(slot.age), 0, 100);
      const int slotPvBonus = (slot.isPV && slot.age == currentAge_) ? 4 : 0;
      const int entryPvBonus = (isEntryPV || entry.isPV) ? 4 : 0;
      const int effectiveOldDepth = slot.depth + slotPvBonus - 3 * ageDiff;

      if (entry.depth + entryPvBonus >= effectiveOldDepth) {
        slot = entry;
        slot.age = currentAge_;
      }
    }

    const Entry* get(ZobristHash::Hash key) const noexcept {
      const auto& entry = entries_[getIndex(key)];
      if (entry.isValid() && entry.key == key) {
        return &entry;
      }
      return nullptr;
    }

    void clear() noexcept {
      currentAge_ = 0;
      std::fill_n(entries_.get(), capacity_, Entry{});
    }
  };
}
