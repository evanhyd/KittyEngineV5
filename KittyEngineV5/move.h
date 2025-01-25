#pragma once
#include <array>
#include <bit>
#include <immintrin.h>

using Bitboard = std::uint64_t;

// Magic holds magic bitboard information for a given square.
struct Magic {
  Bitboard mask;
  Bitboard* attack;
};

enum : size_t {
  A8, B8, C8, D8, E8, F8, G8, H8,
  A7, B7, C7, D7, E7, F7, G7, H7,
  A6, B6, C6, D6, E6, F6, G6, H6,
  A5, B5, C5, D5, E5, F5, G5, H5,
  A4, B4, C4, D4, E4, F4, G4, H4,
  A3, B3, C3, D3, E3, F3, G3, H3,
  A2, B2, C2, D2, E2, F2, G2, H2,
  A1, B1, C1, D1, E1, F1, G1, H1,
};

enum Team : size_t {
  kWhite,
  kBlack
};

constexpr size_t kRankSize = 8;
constexpr size_t kFileSize = 8;
constexpr size_t kSquareSize = kRankSize * kFileSize;
constexpr size_t kTeamSize = 2;

// Chess board rank/file masks.
constexpr Bitboard kFileA = 0x101010101010101ull;
constexpr Bitboard kFileB = kFileA << 1;
constexpr Bitboard kFileC = kFileA << 2;
constexpr Bitboard kFileD = kFileA << 3;
constexpr Bitboard kFileE = kFileA << 4;
constexpr Bitboard kFileF = kFileA << 5;
constexpr Bitboard kFileG = kFileA << 6;
constexpr Bitboard kFileH = kFileA << 7;
constexpr Bitboard kRank8 = 0xffull;
constexpr Bitboard kRank7 = kRank8 << (1 * kRankSize);
constexpr Bitboard kRank6 = kRank8 << (2 * kRankSize);
constexpr Bitboard kRank5 = kRank8 << (3 * kRankSize);
constexpr Bitboard kRank4 = kRank8 << (4 * kRankSize);
constexpr Bitboard kRank3 = kRank8 << (5 * kRankSize);
constexpr Bitboard kRank2 = kRank8 << (6 * kRankSize);
constexpr Bitboard kRank1 = kRank8 << (7 * kRankSize);

// Bit manipulation.
inline constexpr Bitboard setSquare(Bitboard bitboard, size_t square) {
  return bitboard | 1ull << square;
}

inline constexpr Bitboard unsetSquare(Bitboard bitboard, size_t square) {
  return bitboard & ~(1ull << square);
}

inline constexpr Bitboard getSquare(Bitboard bitboard, size_t square) {
  return bitboard >> square & 1;
}

// Generate pawn attack bitboard mask.
template <Team team>
inline constexpr Bitboard generatePawnAttack(size_t square) {
  Bitboard bitboard = setSquare(Bitboard{}, square);
  if constexpr (team == Team::kWhite) {
    return (bitboard >> 9 & ~kFileH) | 
           (bitboard >> 7 & ~kFileA);
  } else if constexpr (team == Team::kBlack) {
    return (bitboard << 7 & ~kFileH) |
           (bitboard << 9 & ~kFileA);
  } else {
    static_assert(false, "invalid team");
  }
}

// Generate knight attack bitboard mask.
inline constexpr Bitboard generateKnightAttack(size_t square) {
  Bitboard bitboard = setSquare(Bitboard{}, square);
  return (bitboard >> 17 & ~kFileH) |
         (bitboard >> 15 & ~kFileA) |
         (bitboard >> 10 & ~(kFileG | kFileH)) |
         (bitboard >> 6 & ~(kFileA | kFileB)) |
         (bitboard << 17 & ~kFileA) |
         (bitboard << 15 & ~kFileH) |
         (bitboard << 10 & ~(kFileA | kFileB)) |
         (bitboard << 6 & ~(kFileG | kFileH));
}

// Generate king attack bitboard mask;
inline constexpr Bitboard generateKingAttack(size_t square) {
  Bitboard bitboard = setSquare(Bitboard{}, square);
  return (bitboard >> 9 & ~kFileH) |
         (bitboard >> 8) |
         (bitboard >> 7 & ~kFileA) |
         (bitboard >> 1 & ~kFileH) |
         (bitboard << 9 & ~kFileA) |
         (bitboard << 8) |
         (bitboard << 7 & ~kFileH) |
         (bitboard << 1 & ~kFileA);    
}

// Attack tables.
constexpr std::array<std::array<Bitboard, kSquareSize>, kTeamSize> kPawnAttackTable = []() {
    std::array<std::array<Bitboard, kSquareSize>, kTeamSize> table{};
    for (size_t i = 0; i < kSquareSize; ++i) {
      table[kWhite][i] = generatePawnAttack<kWhite>(i);
      table[kBlack][i] = generatePawnAttack<kBlack>(i);
    }
    return table;
  }();

constexpr std::array<Bitboard, kSquareSize> kKnightAttackTable = []() {
    std::array<Bitboard, kSquareSize> table{};
    for (size_t i = 0; i < kSquareSize; ++i) {
      table[i] = generateKnightAttack(i);
    }
    return table;
  }();

constexpr std::array<Bitboard, kSquareSize> kKingAttackTable = []() {
    std::array<Bitboard, kSquareSize> table{};
    for (size_t i = 0; i < kSquareSize; ++i) {
      table[i] = generateKingAttack(i);
    }
    return table;
  }();
