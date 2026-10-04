#pragma once
#include "bitboard.h"
#include <array>
#include <cstdint>

namespace bb {
  class ZobristHash {
    uint64_t hash_;

  public:
    static constexpr std::array<std::array<std::array<uint64_t, kSquareSize>, kPieceSize>, kSideSize> kPieceHashes = {

    };

    static constexpr uint64_t kSideToMoveHash = 0;

    static constexpr std::array<uint64_t, 4> kCastleHashes = {};

    static constexpr std::array<uint64_t, kBoardLenSize> kEnpassantHashes = {};

    constexpr void markPiece(Side side, Piece piece, Square square) {
      hash_ ^= kPieceHashes[side][piece][square];
    }

    constexpr void markSide() {
      hash_ ^= kSideToMoveHash;
    }

    constexpr void markCastle(Bitboard oldCastlePermission, Bitboard newCastlePermission) {
      
    }

    constexpr void markEnpassant(Square square) {
      hash_ ^= getSquareFile(square);
    }

    constexpr uint64_t hash() const {
      return hash_;
    }
  };
}

