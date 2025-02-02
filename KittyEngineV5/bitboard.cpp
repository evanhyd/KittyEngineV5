#include "bitboard.h"
#include <iostream>
#include <format>
#include <array>
#include <map>
#include <unordered_map>
#include <bit>
#include <immintrin.h>

// Utility functions/tables.
#define countPiece(bitboard) = std::popcount(bitboard);
#define findFirstPiece(bitboard) std::countr_zero(bitboard);

template <typename ...Squares>
[[nodiscard]] constexpr Bitboard unsetSquare(Bitboard bitboard, Squares... squares) {
  (assert(squares < kSquareSize), ...);
  return bitboard & ~((1ull << squares) | ...);
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

char pieceToAscii(Team team, Piece piece) {
  constexpr std::array<std::array<char, kPieceSize>, kTeamSize> table = { {
    {'P', 'N', 'B', 'R', 'Q', 'K'},
    {'p', 'n', 'b', 'r', 'q', 'k'},
  } };
  return table[team][piece];
}

std::pair<Team, Piece> asciiToPiece(char ascii) {
  for (Team team : {kWhite, kBlack}) {
    for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
      if (pieceToAscii(team, piece) == ascii) {
        return { team, piece };
      }
    }
  }
  assert(false && "invalid ascii value");
  return {}; // unreachable
}

std::string teamToString(Team team) {
  static const std::array<std::string, kTeamSize> table = {
    "white",
    "black",
  };
  return table[team];
}

std::string squareToString(Square square) {
  static const std::array<std::string, kSquareSize + 1> table = {
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "-",
  };
  return table[square];
}

Square squareStringToSquare(const std::string& squareString) {
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    if (squareToString(static_cast<Square>(i)) == squareString) {
      return static_cast<Square>(i);
    }
  }
  assert(false && "invalid square");
  return {};
}

std::string castleToString(CastlePermission permission) {
  static const std::array<std::string, 1ull << 4> table = {
    "-",     // 0B0000
    "K",     // 0B0001
    "Q",     // 0B0010
    "KQ",    // 0B0011
    "k",     // 0B0100
    "Kk",    // 0B0101
    "Qk",    // 0B0110
    "KQk",   // 0B0111
    "q",     // 0B1000
    "Kq",    // 0B1001
    "Qq",    // 0B1010
    "KQq",   // 0B1011
    "kq",    // 0B1100
    "Kkq",   // 0B1101
    "Qkq",   // 0B1110
    "KQkq"   // 0B1111
  };
  return table[permission];
}

void printBitboard(uint64_t bitboard) {
  using std::cout;
  using std::format;
  for (uint32_t i = 0; i < kSideSize; ++i) {
    cout << format("{}|", kSideSize - i);
    for (uint32_t j = 0; j < kSideSize; ++j) {
      cout << format(" {:d}", isSquareSet(bitboard, i * kSideSize + j));
    }
    cout << '\n';
  }
  cout << format("   A B C D E F G H\nBitboard Hex: {:#018x}\n\n", bitboard);
}


// Bitboard masks.
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


// Attack Tables.
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

struct Magic {
  Bitboard  maxAttackNoEdge; // maximum attack pattern excludes the border
  Bitboard* attackReachable; // reachable attack table range, indexed by pext(occupancy, pattern)

  Bitboard getAttack(Bitboard occupancy) const {
    return attackReachable[_pext_u64(occupancy, maxAttackNoEdge)];
  }
};

constexpr Bitboard generateSliderAttackReachable(Piece piece, uint32_t square, Bitboard occupancy) {
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

std::array<Bitboard, 5248> bishopAttackReachableTable{};
std::array<Bitboard, 102400> rookAttackReachableTable{};
const auto magicTable = []() {
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
