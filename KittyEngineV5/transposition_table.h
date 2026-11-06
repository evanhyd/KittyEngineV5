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

      [[nodiscard]] constexpr bool isExact() const noexcept {
        return scoreType == ScoreType::Exact;
      }

      [[nodiscard]] constexpr bool isLowerBound() const noexcept {
        return scoreType == ScoreType::LowerBound;
      }

      [[nodiscard]] constexpr bool isUpperBound() const noexcept {
        return scoreType == ScoreType::UpperBound;
      }

      [[nodiscard]] constexpr bool isValid() const noexcept {
        return scoreType != ScoreType::None;
      }
    };
    static_assert(std::is_aggregate_v<Entry>, "Entry must remain an aggregate for designated initializers");
    static_assert(std::is_trivially_copyable_v<Entry>, "Entry must be trivially copyable");
    static_assert(sizeof(Entry) == 32, "Entry must be exactly 32 bytes to fit into a cache");

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

    [[nodiscard]] constexpr int getEffectiveDepth(const Entry& slot) const noexcept {
      if (!slot.isValid()) {
        return -1000;
      }
      const int ageDiff = static_cast<int>(currentAge_ - slot.age);
      const int pvAndExactBonus = (slot.isPV ? 4 : 0) + (slot.isExact() ? 1 : 0);
      return slot.depth + pvAndExactBonus - 4 * ageDiff;
    }

    void updateSlot(Entry& slot, const Entry& entry) noexcept {
      if ((entry.isExact() && !slot.isExact()) || (entry.isPV && !slot.isPV) || (entry.depth > slot.depth)) {
        slot.score = entry.score;
        slot.depth = entry.depth;
        slot.scoreType = entry.scoreType;
        if (entry.bestMove) {
          slot.bestMove = entry.bestMove;
        }
      } else if (!slot.bestMove) {
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
      slot.isPV |= entry.isPV;
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

    void put(const Entry& entry) noexcept {
      auto& bucket = buckets_[getIndex(entry.key)];
      auto& slot0 = bucket.entries[0];
      auto& slot1 = bucket.entries[1];

      // Update existing entry if present
      if (slot0.isValid() && slot0.key == entry.key) {
        updateSlot(slot0, entry);
        return;
      }
      if (slot1.isValid() && slot1.key == entry.key) {
        updateSlot(slot1, entry);
        return;
      }

      // Pick slot with lower retention value (empty slots have lowest value)
      const int effectiveDepth0 = getEffectiveDepth(slot0);
      const int effectiveDepth1 = getEffectiveDepth(slot1);
      const bool pick0 = (effectiveDepth0 <= effectiveDepth1);
      auto& targetSlot = pick0 ? slot0 : slot1;
      //const int minEffectiveDepth = pick0 ? effectiveDepth0 : effectiveDepth1;
      //const int entryBonus = (entry.isPV ? 4 : 0) + (entry.isExact() ? 1 : 0);
      //if (entry.depth + entryBonus + 3 >= minEffectiveDepth) {
        targetSlot = entry;
        targetSlot.age = currentAge_;
      //}
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
