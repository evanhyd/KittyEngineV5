#pragma once
#include "bitboard.h"
#include <cassert>
#include <immintrin.h>

///////////////////////////////////////////////////////
//                   ATTACK TABLES
///////////////////////////////////////////////////////
template <Piece>
int kAttackTable;

template <>
inline constexpr auto kAttackTable<kPawn> = []() {
  constexpr auto generateAttack = [](Team team, uint32_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    if (team == Team::kWhite) {
      return (shiftUpLeft(bitboard) & ~kFileHMask) | (shiftUpRight(bitboard) & ~kFileAMask);
    } else { // Team::kBlack
      return (shiftDownLeft(bitboard) & ~kFileHMask) | (shiftDownRight(bitboard) & ~kFileAMask);
    }
  };

  std::array<std::array<Bitboard, kSquareSize>, kTeamSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[kWhite][i] = generateAttack(kWhite, i);
    table[kBlack][i] = generateAttack(kBlack, i);
  }
  return table;
}();

template <>
inline constexpr auto kAttackTable<kKnight> = []() {
  constexpr auto generateAttack = [](uint32_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    return (bitboard >> 17 & ~kFileHMask) | (bitboard >> 15 & ~kFileAMask) |
      (bitboard >> 10 & ~(kFileGMask | kFileHMask)) | (bitboard >> 6 & ~(kFileAMask | kFileBMask)) |
      (bitboard << 17 & ~kFileAMask) | (bitboard << 15 & ~kFileHMask) |
      (bitboard << 10 & ~(kFileAMask | kFileBMask)) | (bitboard << 6 & ~(kFileGMask | kFileHMask));
  };

  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

template <>
inline constexpr auto kAttackTable<kKing> = []() {
  constexpr auto generateAttack = [](uint32_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    return (shiftUpLeft(bitboard) & ~kFileHMask) | shiftUp(bitboard) |
      (shiftUpRight(bitboard) & ~kFileAMask) | (shiftLeft(bitboard) & ~kFileHMask) |
      (shiftDownRight(bitboard) & ~kFileAMask) | shiftDown(bitboard) |
      (shiftDownLeft(bitboard) & ~kFileHMask) | (shiftRight(bitboard) & ~kFileAMask);
  };

  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

class SliderTable {
  Bitboard  maxAttackNoEdge; // maximum attack pattern excludes the border
  Bitboard* attackReachable; // reachable attack table range, indexed by pext(occupancy, pattern)

  Bitboard getAttack(Bitboard occupancy) const {
    return attackReachable[_pext_u64(occupancy, maxAttackNoEdge)];
  }

  // Slider Attack Generator
  inline static std::array<Bitboard, 5248> bishopAttackReachableTable{};
  inline static std::array<Bitboard, 102400> rookAttackReachableTable{};
  static const std::array<std::array<SliderTable, 2>, kSquareSize> magicTable;

  static constexpr Bitboard generateSliderAttackReachable(Piece piece, uint32_t square, Bitboard occupancy) {
    constexpr auto isInRange = [](int32_t r, int32_t f) {
      return 0 <= r && r < kSideSize && 0 <= f && f < kSideSize;
    };

    // Cast to int to avoid underflow.
    int32_t rank = static_cast<int32_t>(getSquareRank(square));
    int32_t file = static_cast<int32_t>(getSquareFile(square));
    std::pair<int32_t, int32_t> bishopDir[] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
    std::pair<int32_t, int32_t> rookDir[] = { {1, 0}, {-1, 0}, {0, -1}, {0, 1} };

    Bitboard reachable = 0;
    for (const auto& [dx, dy] : (piece == kBishop ? bishopDir : rookDir)) {
      for (int32_t r = rank + dx, f = file + dy; isInRange(r, f); r += dx, f += dy) {
        reachable = setSquare(reachable, rankFileToSquare(r, f));
        if (isSquareSet(occupancy, rankFileToSquare(r, f))) {
          break;
        }
      }
    }
    return reachable;
  }

  static constexpr std::array<std::array<SliderTable, 2>, kSquareSize> generateMagic() {
    std::array<std::array<SliderTable, 2>, kSquareSize> table{};

    // Generate for both bishop and rook.
    for (Piece piece : {kBishop, kRook}) {
      uint32_t tableIndex = piece - kBishop;
      uint32_t attackTableOffset = 0;

      for (uint32_t i = 0; i < kSquareSize; ++i) {
        Bitboard maxAttack = piece == kBishop ?
          unsetSquare((kSquareToDiagonalMaskTable[i] | kSquareToAntiDiagonalMaskTable[i]), i) :
          unsetSquare((kSquareToRankMaskTable[i] | kSquareToFileMaskTable[i]), i);

        // Remove the edge since the sliding piece must stop at the edge. This reduces the occupancy permutation size.
        Bitboard edge = (kRank1Mask | kRank8Mask) & ~kSquareToRankMaskTable[i] | (kFileAMask | kFileHMask) & ~kSquareToFileMaskTable[i];
        SliderTable& magic = table[i][tableIndex];
        magic.maxAttackNoEdge = maxAttack & ~edge;

        // Assign the attack table segment range related to this square.
        // The range size depends on the possible permutation.
        magic.attackReachable = (piece == kBishop ? bishopAttackReachableTable.data() : rookAttackReachableTable.data()) + attackTableOffset;
        attackTableOffset += 1ull << countPiece(magic.maxAttackNoEdge);

        // Generate all occupancy combination for each attack pattern.
        for (Bitboard occupancy = magic.maxAttackNoEdge; occupancy > 0; occupancy = (occupancy - 1) & magic.maxAttackNoEdge) {
          Bitboard key = _pext_u64(occupancy, magic.maxAttackNoEdge);
          magic.attackReachable[key] = generateSliderAttackReachable(piece, i, occupancy);
        }
      }

      assert(attackTableOffset == (piece == kBishop ? bishopAttackReachableTable.size() : rookAttackReachableTable.size()));
    }
    return table;
  }

public:
  template <Piece piece>
  inline static Bitboard getAttack(uint32_t square, Bitboard occupancy) {
    if constexpr (piece == kBishop) {
      return magicTable[square][0].getAttack(occupancy);
    } else if constexpr (piece == kRook) {
      return magicTable[square][1].getAttack(occupancy);
    } else if constexpr (piece == kQueen) {
      return magicTable[square][0].getAttack(occupancy) | magicTable[square][1].getAttack(occupancy);
    } else {
      static_assert(false, "invalid piece type");
    }
  }
};

inline const std::array<std::array<SliderTable, 2>, kSquareSize> SliderTable::magicTable = SliderTable::generateMagic();
