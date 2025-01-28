#include "bitboard.h"
#include <iostream>
#include <cassert>
#include <array>
#include <bit>
#include <immintrin.h>

// Basic bitboard information.
constexpr size_t kTeamSize = 2;
constexpr size_t kSideSize = 8;
constexpr size_t kDiagonalSize = kSideSize + kSideSize - 1;
constexpr size_t kSquareSize = kSideSize * kSideSize;
constexpr Bitboard kFileAMask = 0x101010101010101ull;
constexpr Bitboard kFileBMask = kFileAMask << 1;
constexpr Bitboard kFileCMask = kFileAMask << 2;
constexpr Bitboard kFileDMask = kFileAMask << 3;
constexpr Bitboard kFileEMask = kFileAMask << 4;
constexpr Bitboard kFileFMask = kFileAMask << 5;
constexpr Bitboard kFileGMask = kFileAMask << 6;
constexpr Bitboard kFileHMask = kFileAMask << 7;
constexpr Bitboard kRank8Mask = 0xffull;
constexpr Bitboard kRank7Mask = kRank8Mask << (1 * kSideSize);
constexpr Bitboard kRank6Mask = kRank8Mask << (2 * kSideSize);
constexpr Bitboard kRank5Mask = kRank8Mask << (3 * kSideSize);
constexpr Bitboard kRank4Mask = kRank8Mask << (4 * kSideSize);
constexpr Bitboard kRank3Mask = kRank8Mask << (5 * kSideSize);
constexpr Bitboard kRank2Mask = kRank8Mask << (6 * kSideSize);
constexpr Bitboard kRank1Mask = kRank8Mask << (7 * kSideSize);

// Utility functions/tables.
[[nodiscard]] constexpr Bitboard setSquare(Bitboard bitboard, size_t square) {
  return bitboard | 1ull << square;
}

[[nodiscard]] constexpr Bitboard unsetSquare(Bitboard bitboard, size_t square) {
  return bitboard & ~(1ull << square);
}

constexpr bool isSquareSet(Bitboard bitboard, size_t square) {
  return bitboard >> square & 1;
}

constexpr size_t getSquareRank(size_t square) {
  return square / kSideSize;
}

constexpr size_t getSquareFile(size_t square) {
  return square % kSideSize; // Compiler optimize it to & 7
}

constexpr size_t toSquareFromRankFile(size_t rank, size_t file) {
  return rank * kSideSize + file;
}

constexpr Bitboard toMaskFromRankFile(size_t rank, size_t file) {
  return 1ull << toSquareFromRankFile(rank, file);
}

void printBitboard(uint64_t bitboard) {
  using std::cout;
  auto fmtFlag = cout.flags();
  for (size_t i = 0; i < kSideSize; ++i) {
    cout << (kSideSize - i) << '|';
    for (size_t j = 0; j < kSideSize; ++j) {
      cout << ' ' << isSquareSet(bitboard, i * kSideSize + j);
    }
    cout << '\n';
  }
  cout << "   A B C D E F G H\nBitboard Hex: " << std::hex << bitboard << "\n\n";
  cout.flags(fmtFlag);
}

constexpr std::array<Bitboard, kDiagonalSize> kDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 1 };
  for (size_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (table[i - 1] << 1 & ~kFileAMask) | (table[i - 1] << kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kDiagonalSize> kAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 0x80 };
  for (size_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (table[i - 1] >> 1 & ~kFileHMask) | (table[i - 1] << kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToRankMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = kRank8Mask << (getSquareRank(i) * kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToFileMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = kFileAMask << getSquareFile(i);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = kDiagonalMaskTable[getSquareRank(i) + getSquareFile(i)];
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = kAntiDiagonalMaskTable[getSquareRank(i) + kSideSize - getSquareFile(i) - 1];
  }
  return table;
}();

// Attack Tables.
struct Magic {
  Bitboard  maxAttackNoEdge; // maximum attack pattern excludes the border
  Bitboard* attackReachable; // reachable attack table range, indexed by pext(occupancy, pattern)

  Bitboard getAttack(Bitboard occupancy) const {
    return attackReachable[_pext_u64(occupancy, maxAttackNoEdge)];
  }
};

constexpr Bitboard generateSliderAttackReachable(size_t piece, size_t square, Bitboard occupancy) {
  constexpr auto isInRange = [](int r, int f) {
    return 0 <= r && r < kSideSize && 0 <= f && f < kSideSize;
  };

  // Cast to int to avoid underflow.
  int rank = static_cast<int>(getSquareRank(square));
  int file = static_cast<int>(getSquareFile(square));
  std::pair<int, int> bishopDir[] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
  std::pair<int, int> rookDir[] = { {1, 0}, {-1, 0}, {0, -1}, {0, 1} };

  Bitboard reachable = 0;
  for (const auto& [dx, dy] : (piece == kBishop ? bishopDir : rookDir)) {
    for (int r = rank + dx, f = file + dy; isInRange(r, f); r += dx, f += dy) {
      reachable = setSquare(reachable, toSquareFromRankFile(r, f));
      if (isSquareSet(occupancy, toSquareFromRankFile(r, f))) {
        break;
      }
    }
  }
  return reachable;
}

constexpr std::array<std::array<Bitboard, kSquareSize>, kTeamSize> kPawnAttackTable = []() {
  constexpr auto generateAttack = [](Team team, size_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    if (team == Team::kWhite) {
      return (bitboard >> 9 & ~kFileHMask) | (bitboard >> 7 & ~kFileAMask);
    } else { // Team::kBlack
      return (bitboard << 7 & ~kFileHMask) | (bitboard << 9 & ~kFileAMask);
    }
  };

  std::array<std::array<Bitboard, kSquareSize>, kTeamSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[kWhite][i] = generateAttack(kWhite, i);
    table[kBlack][i] = generateAttack(kBlack, i);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kKnightAttackTable = []() {
  constexpr auto generateAttack = [](size_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    return (bitboard >> 17 & ~kFileHMask) | (bitboard >> 15 & ~kFileAMask) |
      (bitboard >> 10 & ~(kFileGMask | kFileHMask)) | (bitboard >> 6 & ~(kFileAMask | kFileBMask)) |
      (bitboard << 17 & ~kFileAMask) | (bitboard << 15 & ~kFileHMask) |
      (bitboard << 10 & ~(kFileAMask | kFileBMask)) | (bitboard << 6 & ~(kFileGMask | kFileHMask));
  };

  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kKingAttackTable = []() {
  constexpr auto generateAttack = [](size_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    return (bitboard >> 9 & ~kFileHMask) | (bitboard >> 8) |
      (bitboard >> 7 & ~kFileAMask) | (bitboard >> 1 & ~kFileHMask) |
      (bitboard << 9 & ~kFileAMask) | (bitboard << 8) |
      (bitboard << 7 & ~kFileHMask) | (bitboard << 1 & ~kFileAMask);
  };

  std::array<Bitboard, kSquareSize> table{};
  for (size_t i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

std::array<Bitboard, 5248> bishopAttackReachableTable{};
std::array<Bitboard, 102400> rookAttackReachableTable{};
const std::array<std::array<Magic, 2>, kSquareSize> magicTable = []() {
  std::array<std::array<Magic, 2>, kSquareSize> table{};

  // Generate for both bishop and rook.
  for (size_t piece = kBishop; piece < kRook; ++piece) {
    size_t tableIndex = piece - kBishop;
    size_t attackTableOffset = 0;

    for (size_t i = 0; i < kSquareSize; ++i) {
      Bitboard maxAttack = piece == kBishop ?
        unsetSquare((kSquareToDiagonalMaskTable[i] | kSquareToAntiDiagonalMaskTable[i]), i) :
        unsetSquare((kSquareToRankMaskTable[i] | kSquareToFileMaskTable[i]), i);

      // Remove the edge since the sliding piece must stop at the edge. This reduces the occupancy permutation size.
      Bitboard edge = (kRank1Mask | kRank8Mask) & ~kSquareToRankMaskTable[i] | (kFileAMask | kFileHMask) & ~kSquareToFileMaskTable[i];
      Magic& magic = table[i][tableIndex];
      magic.maxAttackNoEdge = maxAttack & ~edge;

      // Assign the attack table segment range to this square.
      magic.attackReachable = (piece == kBishop ? bishopAttackReachableTable.data() : rookAttackReachableTable.data()) + attackTableOffset;
      attackTableOffset += 1ull << std::popcount(magic.maxAttackNoEdge);

      // Generate all occupancy combination for each attack pattern.
      for (Bitboard occupancy = magic.maxAttackNoEdge; occupancy > 0; occupancy = (occupancy - 1) & magic.maxAttackNoEdge) {
        Bitboard key = _pext_u64(occupancy, magic.maxAttackNoEdge);
        magic.attackReachable[key] = generateSliderAttackReachable(piece, i, occupancy);
      }
    }

    assert(attackTableOffset == (piece == kBishop ? bishopAttackReachableTable.size() : rookAttackReachableTable.size()));
  }

  return table;
}();
