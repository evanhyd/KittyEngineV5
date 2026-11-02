#pragma once
#include "move.h"
#include "zobrist_hash.h"
#include <algorithm>
#include <array>
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
    static constexpr size_t kBucketSize = std::hardware_constructive_interference_size / sizeof(Entry);

    struct alignas(std::hardware_destructive_interference_size) TTBucket {
      std::array<Entry, kBucketSize> entries{};
    };
    static_assert(sizeof(TTBucket) <= std::hardware_constructive_interference_size, "TTBucket must fit into a cache line");

  private:
    const size_t numBuckets_;
    const size_t capacity_;
    std::unique_ptr<TTBucket[]> buckets_;
    uint16_t currentAge_ = 0;

    [[nodiscard]] constexpr size_t getIndex(ZobristHash::Hash key) const noexcept {
      return key & (numBuckets_ - 1);
    }

    [[nodiscard]] int getEffectiveDepth(const Entry& slot) const noexcept {
      if (!slot.isValid()) return -1000;
      const int ageDiff = std::clamp(static_cast<int>(currentAge_) - static_cast<int>(slot.age), 0, 100);
      const int slotPvBonus = (slot.isPV && slot.age == currentAge_) ? 4 : 0;
      return slot.depth + slotPvBonus - 3 * ageDiff;
    }

    template <bool isEntryPV, ScoreType entryScoreType>
    void updateSlot(Entry& slot, const Entry& entry) noexcept {
      const bool isExact = (entryScoreType == ScoreType::Exact) || (entry.scoreType == ScoreType::Exact);
      const bool entryIsPV = isEntryPV || entry.isPV;
      const int pvBonus = entryIsPV ? (slot.isPV ? 0 : -1) : (slot.isPV ? 4 : 0);

      if ((isExact && slot.scoreType != ScoreType::Exact) || (entry.depth >= slot.depth + pvBonus)) {
        const Move prevMove = slot.bestMove;
        slot = entry;
        if (slot.bestMove == Move{}) slot.bestMove = prevMove;
      } else if (slot.bestMove == Move{}) {
        slot.bestMove = entry.bestMove;
      }

      if (entry.hasStaticScore) {
        slot.staticScore = entry.staticScore;
        slot.hasStaticScore = true;
      }
      if (entry.hasInCheck) {
        slot.inCheck = entry.inCheck;
        slot.hasInCheck = true;
      }
      slot.isPV |= entryIsPV;
      slot.age = currentAge_;
    }

  public:
    explicit TranspositionTable(size_t capacity)
      : numBuckets_(capacity / kBucketSize),
        capacity_(capacity) {
      if (!std::has_single_bit(capacity) || capacity < kBucketSize) {
        throw std::invalid_argument("transposition table entry size must be a power of two and at least bucket size");
      }
      buckets_ = std::make_unique<TTBucket[]>(numBuckets_);
    }

    [[nodiscard]] size_t capacity() const noexcept {
      return capacity_;
    }

    void age() noexcept {
      ++currentAge_;
    }

    void prefetch(ZobristHash::Hash key) noexcept {
      _mm_prefetch(reinterpret_cast<const char*>(&buckets_[getIndex(key)]), _MM_HINT_T0);
    }

    template <bool isEntryPV = false, ScoreType entryScoreType = ScoreType::None>
    void put(const Entry& entry) noexcept {
      auto& bucket = buckets_[getIndex(entry.key)];
      auto& slot0 = bucket.entries[0];
      auto& slot1 = bucket.entries[1];

      // Update existing entry if present
      if (slot0.isValid() && slot0.key == entry.key) {
        updateSlot<isEntryPV, entryScoreType>(slot0, entry);
        return;
      }
      if (slot1.isValid() && slot1.key == entry.key) {
        updateSlot<isEntryPV, entryScoreType>(slot1, entry);
        return;
      }

      // Pick slot with lower retention value (empty slots have lowest value)
      const int effectiveDepth0 = getEffectiveDepth(slot0);
      const int effectiveDepth1 = getEffectiveDepth(slot1);
      auto& targetSlot = (effectiveDepth0 <= effectiveDepth1) ? slot0 : slot1;

      const int entryPvBonus = (isEntryPV || entry.isPV) ? 4 : 0;
      if (entry.depth + entryPvBonus >= std::min(effectiveDepth0, effectiveDepth1)) {
        targetSlot = entry;
        targetSlot.age = currentAge_;
      }
    }

    [[nodiscard]] const Entry* get(ZobristHash::Hash key) const noexcept {
      const auto& bucket = buckets_[getIndex(key)];
      if (bucket.entries[0].isValid() && bucket.entries[0].key == key) {
        return &bucket.entries[0];
      }
      if (bucket.entries[1].isValid() && bucket.entries[1].key == key) {
        return &bucket.entries[1];
      }
      return nullptr;
    }

    void clear() noexcept {
      currentAge_ = 0;
      std::fill_n(buckets_.get(), numBuckets_, TTBucket{});
    }
  };
}
