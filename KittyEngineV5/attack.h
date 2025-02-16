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

namespace internal {
  inline constexpr Bitboard generateSliderAttackReachable(Piece piece, uint32_t square, Bitboard occupancy) {
    constexpr auto isInRange = [](int32_t r, int32_t f) {
      return 0 <= r && r < kSideSize && 0 <= f && f < kSideSize;
    };

    // Cast to int to avoid underflow.
    int32_t rank = static_cast<int32_t>(getSquareRank(square));
    int32_t file = static_cast<int32_t>(getSquareFile(square));
    constexpr std::pair<int32_t, int32_t> bishopDir[] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
    constexpr std::pair<int32_t, int32_t> rookDir[] = { {1, 0}, {-1, 0}, {0, -1}, {0, 1} };

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
}

// Use PEXT intrinsic to map the bitboards
// Avoid using PEXT on AMD due to slow implementation.
//#define USE_PEXT
class SliderTable {
#ifndef USE_PEXT
  Bitboard magicNum;          // Magic bitboard hashing
#endif
  Bitboard mask;              // Maximum attack pattern excludes the border
  Bitboard* attackReachable;  // Reachable attack table range, indexed by magic shifting

  template <Piece piece>
  size_t getKey(Bitboard occupancy) const {
#ifdef USE_PEXT
    return _pext_u64(occupancy, mask);
#else
    constexpr size_t relevantBitsInverse = (piece == kBishop ? kSquareSize - 9 : kSquareSize - 12);
    return ((occupancy | mask) * magicNum) >> relevantBitsInverse;
#endif
  }

  template <Piece piece>
  Bitboard getAttack(Bitboard occupancy) const {
    return attackReachable[getKey<piece>(occupancy)];
  }

#ifdef USE_PEXT
  inline static std::array<Bitboard, 0x19000 + 0x1480> sliderReachableTable{};
#else
  inline static std::array<Bitboard, 87988> sliderReachableTable{};
#endif

  static const std::array<std::array<SliderTable, 2>, kSquareSize> magicTable;

  static constexpr auto generateMagic() {
#ifndef USE_PEXT
    struct BlackMagicConfig {
      Bitboard magic;
      size_t index;
    };
    constexpr std::array<std::array<BlackMagicConfig, kSquareSize>, kTeamSize> blackMagicConfig = { {
      {
        BlackMagicConfig{ 0xa7020080601803d8ull, 60984 },
        BlackMagicConfig{ 0x13802040400801f1ull, 66046 },
        BlackMagicConfig{ 0x0a0080181001f60cull, 32910 },
        BlackMagicConfig{ 0x1840802004238008ull, 16369 },
        BlackMagicConfig{ 0xc03fe00100000000ull, 42115 },
        BlackMagicConfig{ 0x24c00bffff400000ull,   835 },
        BlackMagicConfig{ 0x0808101f40007f04ull, 18910 },
        BlackMagicConfig{ 0x100808201ec00080ull, 25911 },
        BlackMagicConfig{ 0xffa2feffbfefb7ffull, 63301 },
        BlackMagicConfig{ 0x083e3ee040080801ull, 16063 },
        BlackMagicConfig{ 0xc0800080181001f8ull, 17481 },
        BlackMagicConfig{ 0x0440007fe0031000ull, 59361 },
        BlackMagicConfig{ 0x2010007ffc000000ull, 18735 },
        BlackMagicConfig{ 0x1079ffe000ff8000ull, 61249 },
        BlackMagicConfig{ 0x3c0708101f400080ull, 68938 },
        BlackMagicConfig{ 0x080614080fa00040ull, 61791 },
        BlackMagicConfig{ 0x7ffe7fff817fcff9ull, 21893 },
        BlackMagicConfig{ 0x7ffebfffa01027fdull, 62068 },
        BlackMagicConfig{ 0x53018080c00f4001ull, 19829 },
        BlackMagicConfig{ 0x407e0001000ffb8aull, 26091 },
        BlackMagicConfig{ 0x201fe000fff80010ull, 15815 },
        BlackMagicConfig{ 0xffdfefffde39ffefull, 16419 },
        BlackMagicConfig{ 0xcc8808000fbf8002ull, 59777 },
        BlackMagicConfig{ 0x7ff7fbfff8203fffull, 16288 },
        BlackMagicConfig{ 0x8800013e8300c030ull, 33235 },
        BlackMagicConfig{ 0x0420009701806018ull, 15459 },
        BlackMagicConfig{ 0x7ffeff7f7f01f7fdull, 15863 },
        BlackMagicConfig{ 0x8700303010c0c006ull, 75555 },
        BlackMagicConfig{ 0xc800181810606000ull, 79445 },
        BlackMagicConfig{ 0x20002038001c8010ull, 15917 },
        BlackMagicConfig{ 0x087ff038000fc001ull,  8512 },
        BlackMagicConfig{ 0x00080c0c00083007ull, 73069 },
        BlackMagicConfig{ 0x00000080fc82c040ull, 16078 },
        BlackMagicConfig{ 0x000000407e416020ull, 19168 },
        BlackMagicConfig{ 0x00600203f8008020ull, 11056 },
        BlackMagicConfig{ 0xd003fefe04404080ull, 62544 },
        BlackMagicConfig{ 0xa00020c018003088ull, 80477 },
        BlackMagicConfig{ 0x7fbffe700bffe800ull, 75049 },
        BlackMagicConfig{ 0x107ff00fe4000f90ull, 32947 },
        BlackMagicConfig{ 0x7f8fffcff1d007f8ull, 59172 },
        BlackMagicConfig{ 0x0000004100f88080ull, 55845 },
        BlackMagicConfig{ 0x00000020807c4040ull, 61806 },
        BlackMagicConfig{ 0x00000041018700c0ull, 73601 },
        BlackMagicConfig{ 0x0010000080fc4080ull, 15546 },
        BlackMagicConfig{ 0x1000003c80180030ull, 45243 },
        BlackMagicConfig{ 0xc10000df80280050ull, 20333 },
        BlackMagicConfig{ 0xffffffbfeff80fdcull, 33402 },
        BlackMagicConfig{ 0x000000101003f812ull, 25917 },
        BlackMagicConfig{ 0x0800001f40808200ull, 32875 },
        BlackMagicConfig{ 0x084000101f3fd208ull,  4639 },
        BlackMagicConfig{ 0x080000000f808081ull, 17077 },
        BlackMagicConfig{ 0x0004000008003f80ull, 62324 },
        BlackMagicConfig{ 0x08000001001fe040ull, 18159 },
        BlackMagicConfig{ 0x72dd000040900a00ull, 61436 },
        BlackMagicConfig{ 0xfffffeffbfeff81dull, 57073 },
        BlackMagicConfig{ 0xcd8000200febf209ull, 61025 },
        BlackMagicConfig{ 0x100000101ec10082ull, 81259 },
        BlackMagicConfig{ 0x7fbaffffefe0c02full, 64083 },
        BlackMagicConfig{ 0x7f83fffffff07f7full, 56114 },
        BlackMagicConfig{ 0xfff1fffffff7ffc1ull, 57058 },
        BlackMagicConfig{ 0x0878040000ffe01full, 58912 },
        BlackMagicConfig{ 0x945e388000801012ull, 22194 },
        BlackMagicConfig{ 0x0840800080200fdaull, 70880 },
        BlackMagicConfig{ 0x100000c05f582008ull, 11140 }
      },
      {
        BlackMagicConfig{ 0x80280013ff84ffffull, 10890 },
        BlackMagicConfig{ 0x5ffbfefdfef67fffull, 50579 },
        BlackMagicConfig{ 0xffeffaffeffdffffull, 62020 },
        BlackMagicConfig{ 0x003000900300008aull, 67322 },
        BlackMagicConfig{ 0x0050028010500023ull, 80251 },
        BlackMagicConfig{ 0x0020012120a00020ull, 58503 },
        BlackMagicConfig{ 0x0030006000c00030ull, 51175 },
        BlackMagicConfig{ 0x0058005806b00002ull, 83130 },
        BlackMagicConfig{ 0x7fbff7fbfbeafffcull, 50430 },
        BlackMagicConfig{ 0x0000140081050002ull, 21613 },
        BlackMagicConfig{ 0x0000180043800048ull, 72625 },
        BlackMagicConfig{ 0x7fffe800021fffb8ull, 80755 },
        BlackMagicConfig{ 0xffffcffe7fcfffafull, 69753 },
        BlackMagicConfig{ 0x00001800c0180060ull, 26973 },
        BlackMagicConfig{ 0x4f8018005fd00018ull, 84972 },
        BlackMagicConfig{ 0x0000180030620018ull, 31958 },
        BlackMagicConfig{ 0x00300018010c0003ull, 69272 },
        BlackMagicConfig{ 0x0003000c0085ffffull, 48372 },
        BlackMagicConfig{ 0xfffdfff7fbfefff7ull, 65477 },
        BlackMagicConfig{ 0x7fc1ffdffc001fffull, 43972 },
        BlackMagicConfig{ 0xfffeffdffdffdfffull, 57154 },
        BlackMagicConfig{ 0x7c108007befff81full, 53521 },
        BlackMagicConfig{ 0x20408007bfe00810ull, 30534 },
        BlackMagicConfig{ 0x0400800558604100ull, 16548 },
        BlackMagicConfig{ 0x0040200010080008ull, 46407 },
        BlackMagicConfig{ 0x0010020008040004ull, 11841 },
        BlackMagicConfig{ 0xfffdfefff7fbfff7ull, 21112 },
        BlackMagicConfig{ 0xfebf7dfff8fefff9ull, 44214 },
        BlackMagicConfig{ 0xc00000ffe001ffe0ull, 57925 },
        BlackMagicConfig{ 0x4af01f00078007c3ull, 29574 },
        BlackMagicConfig{ 0xbffbfafffb683f7full, 17309 },
        BlackMagicConfig{ 0x0807f67ffa102040ull, 40143 },
        BlackMagicConfig{ 0x200008e800300030ull, 64659 },
        BlackMagicConfig{ 0x0000008780180018ull, 70469 },
        BlackMagicConfig{ 0x0000010300180018ull, 62917 },
        BlackMagicConfig{ 0x4000008180180018ull, 60997 },
        BlackMagicConfig{ 0x008080310005fffaull, 18554 },
        BlackMagicConfig{ 0x4000188100060006ull, 14385 },
        BlackMagicConfig{ 0xffffff7fffbfbfffull,     0 },
        BlackMagicConfig{ 0x0000802000200040ull, 38091 },
        BlackMagicConfig{ 0x20000202ec002800ull, 25122 },
        BlackMagicConfig{ 0xfffff9ff7cfff3ffull, 60083 },
        BlackMagicConfig{ 0x000000404b801800ull, 72209 },
        BlackMagicConfig{ 0x2000002fe03fd000ull, 67875 },
        BlackMagicConfig{ 0xffffff6ffe7fcffdull, 56290 },
        BlackMagicConfig{ 0xbff7efffbfc00fffull, 43807 },
        BlackMagicConfig{ 0x000000100800a804ull, 73365 },
        BlackMagicConfig{ 0x6054000a58005805ull, 76398 },
        BlackMagicConfig{ 0x0829000101150028ull, 20024 },
        BlackMagicConfig{ 0x00000085008a0014ull,  9513 },
        BlackMagicConfig{ 0x8000002b00408028ull, 24324 },
        BlackMagicConfig{ 0x4000002040790028ull, 22996 },
        BlackMagicConfig{ 0x7800002010288028ull, 23213 },
        BlackMagicConfig{ 0x0000001800e08018ull, 56002 },
        BlackMagicConfig{ 0xa3a80003f3a40048ull, 22809 },
        BlackMagicConfig{ 0x2003d80000500028ull, 44545 },
        BlackMagicConfig{ 0xfffff37eefefdfbeull, 36072 },
        BlackMagicConfig{ 0x40000280090013c1ull,  4750 },
        BlackMagicConfig{ 0xbf7ffeffbffaf71full,  6014 },
        BlackMagicConfig{ 0xfffdffff777b7d6eull, 36054 },
        BlackMagicConfig{ 0x48300007e8080c02ull, 78538 },
        BlackMagicConfig{ 0xafe0000fff780402ull, 28745 },
        BlackMagicConfig{ 0xee73fffbffbb77feull,  8555 },
        BlackMagicConfig{ 0x0002000308482882ull,  1009 }
      }
    }};

#endif

    std::array<std::array<SliderTable, kTeamSize>, kSquareSize> table{};

#ifdef USE_PEXT
    size_t offset = 0;
#endif

    // Generate for both bishop and rook.
    for (Piece piece : {kBishop, kRook}) {
      for (uint32_t i = 0; i < kSquareSize; ++i) {
        // Remove the edge since the sliding piece must stop at the edge. This reduces the occupancy permutation size.
        SliderTable& magic = table[i][piece - kBishop];
        Bitboard edge = ((kRank1Mask | kRank8Mask) & ~kSquareToRankMaskTable[i]) | ((kFileAMask | kFileHMask) & ~kSquareToFileMaskTable[i]);

#ifdef USE_PEXT
        magic.mask = internal::generateSliderAttackReachable(piece, i, 0) & ~edge;
        size_t permutation = 1ull << countPiece(magic.mask);
        magic.attackReachable = sliderReachableTable.data() + offset;

        // Generate all occupancy combination for each attack pattern.
        for (Bitboard occupancy = magic.mask; ; occupancy = (occupancy - 1) & magic.mask) {
          size_t key = (piece == kBishop ? magic.getKey<kBishop>(occupancy) : magic.getKey<kRook>(occupancy));
          magic.attackReachable[key] = internal::generateSliderAttackReachable(piece, i, occupancy);
          assert(key < permutation);
          if (occupancy == 0) {
            break;
          }
        }
        offset += permutation;
#else
        // Set up magic bitboard hashing factors.
        magic.magicNum = blackMagicConfig[piece - kBishop][i].magic;

        const Bitboard maxAttackNoEdge = internal::generateSliderAttackReachable(piece, i, 0) & ~edge;
        magic.mask = ~maxAttackNoEdge;
        magic.attackReachable = sliderReachableTable.data() + blackMagicConfig[piece - kBishop][i].index;

        // Generate all occupancy combination for each attack pattern.
        for (Bitboard occupancy = maxAttackNoEdge; ; occupancy = (occupancy - 1) & maxAttackNoEdge) {
          size_t key = (piece == kBishop ? magic.getKey<kBishop>(occupancy) : magic.getKey<kRook>(occupancy));
          magic.attackReachable[key] = internal::generateSliderAttackReachable(piece, i, occupancy);
          assert(key < sliderReachableTable.size());
          if (occupancy == 0) {
            break;
          }
        }
#endif
      }
    }
    return table;
  }

public:
  template <Piece piece>
  inline static Bitboard getAttack(uint32_t square, Bitboard occupancy) {
    if constexpr (piece == kBishop) {
      return magicTable[square][0].getAttack<piece>(occupancy);
    } else if constexpr (piece == kRook) {
      return magicTable[square][1].getAttack<piece>(occupancy);
    } else if constexpr (piece == kQueen) {
      return magicTable[square][0].getAttack<kBishop>(occupancy) | magicTable[square][1].getAttack<kRook>(occupancy);
    } else {
      static_assert(false, "invalid piece type");
    }
  }
};

inline const std::array<std::array<SliderTable, 2>, kSquareSize> SliderTable::magicTable = SliderTable::generateMagic();
