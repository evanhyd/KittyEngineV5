#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <string>
#include <immintrin.h>

using Bitboard = uint64_t;

enum Square : uint32_t {
  A8, B8, C8, D8, E8, F8, G8, H8,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A1, B1, C1, D1, E1, F1, G1, H1,
  NO_SQUARE,
};

enum Team : uint32_t {
  kWhite,
  kBlack,
};

enum Piece : uint32_t {
  kPawn,
  kKnight,
  kBishop,
  kRook,
  kQueen,
  kKing,
};

enum CastlePermission : uint32_t {
  kWhiteKingCastle = 0b0001,
  kWhiteQueenCastle = 0b0010,
  kBlackKingCastle = 0b0100,
  kBlackQueenCastle = 0b1000,
};


///////////////////////////////////////////////////////
//                 UI FORMATTING
///////////////////////////////////////////////////////
char pieceToAscii(Team team, Piece piece);
std::pair<Team, Piece> asciiToPiece(char ascii);
std::string teamToString(Team team);
std::string squareToString(Square square);
Square squareStringToSquare(const std::string& squareString);
std::string castleToString(CastlePermission permission);
void printBitboard(uint64_t bitboard);


///////////////////////////////////////////////////////
//                 BITBOARD MASKS
///////////////////////////////////////////////////////
constexpr uint32_t kTeamSize = 2;
constexpr uint32_t kPieceSize = 6;
constexpr uint32_t kSideSize = 8;
constexpr uint32_t kSquareSize = kSideSize * kSideSize;
constexpr uint32_t kDiagonalSize = kSideSize + kSideSize - 1;
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

constexpr int countPiece(Bitboard bitboard) {
  return std::popcount(bitboard);
}

constexpr uint32_t findFirstPiece(Bitboard bitboard) {
  return static_cast<uint32_t>(std::countr_zero(bitboard));
}

[[nodiscard]] constexpr Bitboard removeFirstPiece(Bitboard bitboard) {
  assert(bitboard != 0);
  return bitboard & (bitboard - 1);
}

template <typename ...Squares>
[[nodiscard]] constexpr Bitboard setSquare(Bitboard bitboard, Squares... squares) {
  (assert(squares < kSquareSize), ...);
  return bitboard | ((1ull << squares) | ...);
}

template <typename ...Squares>
[[nodiscard]] constexpr Bitboard unsetSquare(Bitboard bitboard, Squares... squares) {
  (assert(squares < kSquareSize), ...);
  return bitboard & ~((1ull << squares) | ...);
}

constexpr bool isSquareSet(Bitboard bitboard, uint32_t square) {
  assert(square < kSquareSize);
  return bitboard >> square & 1;
}

constexpr uint32_t getSquareRank(uint32_t square) {
  assert(square < kSquareSize);
  return square / kSideSize;
}

constexpr uint32_t getSquareFile(uint32_t square) {
  assert(square < kSquareSize);
  return square % kSideSize; // Compiler optimize it to & 7
}

constexpr uint32_t toSquareFromRankFile(uint32_t rank, uint32_t file) {
  assert(rank < kSideSize && file < kSideSize);
  return rank * kSideSize + file;
}

constexpr std::array<Bitboard, kDiagonalSize> kDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 1 };
  for (uint32_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (table[i - 1] << 1 & ~kFileAMask) | (table[i - 1] << kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kDiagonalSize> kAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 0x80 };
  for (uint32_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (table[i - 1] >> 1 & ~kFileHMask) | (table[i - 1] << kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToRankMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kRank8Mask << (getSquareRank(i) * kSideSize);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToFileMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kFileAMask << getSquareFile(i);
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kDiagonalMaskTable[getSquareRank(i) + getSquareFile(i)];
  }
  return table;
}();

constexpr std::array<Bitboard, kSquareSize> kSquareToAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kAntiDiagonalMaskTable[getSquareRank(i) + kSideSize - getSquareFile(i) - 1];
  }
  return table;
}();


///////////////////////////////////////////////////////
//                   ATTACK TABLES
///////////////////////////////////////////////////////
constexpr auto kPawnAttackTable = []() {
  constexpr auto generateAttack = [](Team team, uint32_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    if (team == Team::kWhite) {
      return (bitboard >> 9 & ~kFileHMask) | (bitboard >> 7 & ~kFileAMask);
    } else { // Team::kBlack
      return (bitboard << 7 & ~kFileHMask) | (bitboard << 9 & ~kFileAMask);
    }
  };

  std::array<std::array<Bitboard, kSquareSize>, kTeamSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[kWhite][i] = generateAttack(kWhite, i);
    table[kBlack][i] = generateAttack(kBlack, i);
  }
  return table;
}();

constexpr auto kKnightAttackTable = []() {
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

constexpr auto kKingAttackTable = []() {
  constexpr auto generateAttack = [](uint32_t square) {
    Bitboard bitboard = setSquare(Bitboard{}, square);
    return (bitboard >> 9 & ~kFileHMask) | (bitboard >> 8) |
      (bitboard >> 7 & ~kFileAMask) | (bitboard >> 1 & ~kFileHMask) |
      (bitboard << 9 & ~kFileAMask) | (bitboard << 8) |
      (bitboard << 7 & ~kFileHMask) | (bitboard << 1 & ~kFileAMask);
  };

  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

class Magic {
  Bitboard  maxAttackNoEdge; // maximum attack pattern excludes the border
  Bitboard* attackReachable; // reachable attack table range, indexed by pext(occupancy, pattern)

  Bitboard getAttack(Bitboard occupancy) const {
    return attackReachable[_pext_u64(occupancy, maxAttackNoEdge)];
  }

  // Slider Attack Generator
  inline static std::array<Bitboard, 5248> bishopAttackReachableTable{};
  inline static std::array<Bitboard, 102400> rookAttackReachableTable{};
  static const std::array<std::array<Magic, 2>, kSquareSize> magicTable;

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
        reachable = setSquare(reachable, toSquareFromRankFile(r, f));
        if (isSquareSet(occupancy, toSquareFromRankFile(r, f))) {
          break;
        }
      }
    }
    return reachable;
  }

  static constexpr std::array<std::array<Magic, 2>, kSquareSize> generateMagic() {
    std::array<std::array<Magic, 2>, kSquareSize> table{};

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
        Magic& magic = table[i][tableIndex];
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
  inline static Bitboard getBishopAttack(uint32_t square, Bitboard occupancy) {
    return magicTable[square][0].getAttack(occupancy);
  }

  inline static Bitboard getRookAttack(uint32_t square, Bitboard occupancy) {
    return magicTable[square][1].getAttack(occupancy);
  }
};

inline const std::array<std::array<Magic, 2>, kSquareSize> Magic::magicTable = Magic::generateMagic();
