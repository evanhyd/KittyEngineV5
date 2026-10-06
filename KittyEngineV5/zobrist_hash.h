#pragma once
#include "bitboard.h"
#include <array>
#include <cstdint>
#include <algorithm>

namespace bb {

  // Known bug: Two boards have the same pieces set up, one has enpassant square, one does not.
  // They have different hashes, however, they count toward the same 3-fold repetition states.
  class ZobristHash {
  public:
    using Hash = uint64_t;

    constexpr void set(Hash hash) noexcept {
      hash_ = hash;
    }

    constexpr Hash hash() const noexcept {
      return hash_;
    }

    constexpr void reset() noexcept {
      hash_ = 0;
    }

    constexpr void markPiece(Side side, Piece piece, Square square) noexcept {
      hash_ ^= kPieceHashes[side][piece][square];
    }

    constexpr void markSide() noexcept {
      hash_ ^= kSideToMoveHash;
    }

    constexpr void markCastle(CastlePermission permission) noexcept {
      hash_ ^= kCastleHashes[permission];
    }

    constexpr void markEnpassant(Square square) noexcept {
      if (square != NoSquare) {
        hash_ ^= kEnpassantHashes[getSquareFile(square)];
      }
    }

  private:
    Hash hash_;

    struct SplitMix64 {
      Hash state = 0;
      constexpr Hash next() noexcept {
        state += 0x9e3779b97f4a7c15;
        Hash z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
      }
    };

    inline static SplitMix64 generator{};

    inline static const auto kPieceHashes = []() {
      std::array<std::array<std::array<Hash, kSquareSize>, kPieceSize>, kSideSize> table{};
      for (auto& row : table) {
        for (auto& col : row) {
          for (auto& depth : col) {
            depth = generator.next();
          }
        }
      }
      return table;
    }();

    inline static const Hash kSideToMoveHash = generator.next();

    inline static const auto kCastleHashes = []() {
      std::array<Hash, 1 << 4> table{};
      for (auto& value : table) {
        value = generator.next();
      }
      return table;
    }();

    inline static const auto kEnpassantHashes = []() {
      std::array<Hash, kBoardLenSize> table{};
      for (auto& value : table) {
        value = generator.next();
      }
      return table;
    }();
  };
}
