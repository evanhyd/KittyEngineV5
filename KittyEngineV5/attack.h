#pragma once
#include "bitboard.h"
#include <immintrin.h>

///////////////////////////////////////////////////////
//                   ATTACK TABLES
///////////////////////////////////////////////////////
template <Piece>
int kAttackTable;

template <>
inline constexpr auto kAttackTable<kPawn> = []() {
  constexpr auto generateAttack = [](Color color, Square square) {
    Bitboard bitboard = toBitboard(square);
    if (color == kWhite) {
      return shiftUpLeft(bitboard) | shiftUpRight(bitboard);
    } else {
      return shiftDownLeft(bitboard) | shiftDownRight(bitboard);
    }
  };

  std::array<std::array<Bitboard, kSquareSize>, kColorSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[kWhite][i] = generateAttack(kWhite, i);
    table[kBlack][i] = generateAttack(kBlack, i);
  }
  return table;
}();

template <>
inline constexpr auto kAttackTable<kKnight> = []() {
  constexpr auto generateAttack = [](Square square) {
    Bitboard bitboard = toBitboard(square);
    return (bitboard >> 17 & ~kFileHMask) | (bitboard >> 15 & ~kFileAMask) |
      (bitboard >> 10 & ~(kFileGMask | kFileHMask)) | (bitboard >> 6 & ~(kFileAMask | kFileBMask)) |
      (bitboard << 17 & ~kFileAMask) | (bitboard << 15 & ~kFileHMask) |
      (bitboard << 10 & ~(kFileAMask | kFileBMask)) | (bitboard << 6 & ~(kFileGMask | kFileHMask));
  };

  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

template <>
inline constexpr auto kAttackTable<kKing> = []() {
  constexpr auto generateAttack = [](Square square) {
    Bitboard bitboard = toBitboard(square);
    return shiftUpLeft(bitboard) | shiftUp(bitboard) | shiftUpRight(bitboard) | shiftLeft(bitboard) |
           shiftDownRight(bitboard) | shiftDown(bitboard) | shiftDownLeft(bitboard) | shiftRight(bitboard);
  };

  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = generateAttack(i);
  }
  return table;
}();

namespace internal {
  inline constexpr Bitboard generateSliderAttackReachable(Piece piece, Square square, Bitboard occupancy) {
    constexpr auto isInRange = [](int32_t r, int32_t f) {
      return 0 <= r && r < kSideSize && 0 <= f && f < kSideSize;
    };

    // Cast to int to avoid underflow.
    int32_t rank = static_cast<int32_t>(getSquareRank(square));
    int32_t file = static_cast<int32_t>(getSquareFile(square));
    constexpr std::pair<int32_t, int32_t> bishopDir[] = { {int32_t(1), int32_t(1)}, {int32_t(1), int32_t(-1)}, {int32_t(-1), int32_t(1)}, {int32_t(-1), int32_t(-1)} };
    constexpr std::pair<int32_t, int32_t> rookDir[] = { {int32_t(1), int32_t(0)}, {int32_t(-1), int32_t(0)}, {int32_t(0), int32_t(-1)}, {int32_t(0), int32_t(1)} };

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
  Bitboard maxAttackNoEdge;   // Maximum attack pattern excludes the border
  Bitboard* attackReachable;  // Reachable attack table range, indexed by magic shifting

  template <Piece piece>
  size_t getKey(Bitboard occupancy) const { 
#ifdef USE_PEXT
    return _pext_u64(occupancy, maxAttackNoEdge);
#else
    constexpr size_t relevantBitsInverse = (piece == kBishop ? kSquareSize - 9 : kSquareSize - 12);
    return ((occupancy & maxAttackNoEdge) * magicNum) >> relevantBitsInverse;
#endif
  }

  template <Piece piece>
  Bitboard getAttack(Bitboard occupancy) const { return attackReachable[getKey<piece>(occupancy)]; }

#ifdef USE_PEXT
  inline static std::array<Bitboard, 5248> bishopAttackReachableTable{};
  inline static std::array<Bitboard, 102400> rookAttackReachableTable{};
#else
  inline static std::array<Bitboard, 64 * 512> bishopAttackReachableTable{};
  inline static std::array<Bitboard, 64 * 4096> rookAttackReachableTable{};
#endif

  static const std::array<std::array<SliderTable, 2>, kSquareSize> magicTable;

  static constexpr auto generateMagic() {
#ifndef USE_PEXT
    constexpr std::array<std::array<Bitboard, kSquareSize>, kColorSize> kMagicNumTable = { {
      {
        0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
        0x581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
        0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
        0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
        0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
        0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
        0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
        0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
        0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
        0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
        0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
        0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
        0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
        0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
        0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
        0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
      },
      {
        0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL, 0x100081001000420ULL,
        0x200020010080420ULL, 0x3001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
        0x800098204000ULL, 0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
        0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL, 0x801000060821100ULL,
        0x80044006422000ULL, 0x100808020004000ULL, 0x12108a0010204200ULL, 0x140848010000802ULL,
        0x481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
        0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL, 0x20100080080080ULL,
        0x2000a00200410ULL, 0x20080800400ULL, 0x80088400100102ULL, 0x80004600042881ULL,
        0x4040008040800020ULL, 0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
        0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL, 0x200046502000484ULL,
        0x480400080088020ULL, 0x1000422010034000ULL, 0x30200100110040ULL, 0x100021010009ULL,
        0x2002080100110004ULL, 0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
        0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL, 0x2060820c0120200ULL,
        0x1001004080100ULL, 0x20c020080040080ULL, 0x2935610830022400ULL, 0x44440041009200ULL,
        0x280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
        0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL, 0x1004081002402ULL
      }
    }};
#endif

    std::array<std::array<SliderTable, kColorSize>, kSquareSize> table{};

    // Generate for both bishop and rook.
    for (Piece piece : {kBishop, kRook}) {
      size_t offset = 0;

      for (Square i = 0; i < kSquareSize; ++i) {
        SliderTable& magic = table[i][piece - kBishop];

#ifndef USE_PEXT
        // Set up magic bitboard hashing factors.
        magic.magicNum = kMagicNumTable[piece - kBishop][i];
#endif

        // Remove the edge since the sliding piece must stop at the edge. This reduces the occupancy permutation size.
        Bitboard edge = ((kRank1Mask | kRank8Mask) & ~kSquareToRankMasks[i]) | ((kFileAMask | kFileHMask) & ~kSquareToFileMasks[i]);
        magic.maxAttackNoEdge = internal::generateSliderAttackReachable(piece, i, 0) & ~edge;

        // Assign the attack table range.
        magic.attackReachable = (piece == kBishop ? bishopAttackReachableTable.data() : rookAttackReachableTable.data()) + offset;

#ifdef USE_PEXT
        size_t permutations = 1ull << countPiece(magic.maxAttackNoEdge);
#else
        size_t permutations = (piece == kBishop ? (1ull << 9) : (1ull << 12));
#endif
        offset += permutations;

        // Generate all occupancy combination for each attack pattern.
        for (Bitboard occupancy = magic.maxAttackNoEdge; ; occupancy = (occupancy - 1) & magic.maxAttackNoEdge) {
          size_t key = (piece == kBishop ? magic.getKey<kBishop>(occupancy) : magic.getKey<kRook>(occupancy));
          magic.attackReachable[key] = internal::generateSliderAttackReachable(piece, i, occupancy);
          assert(key < permutations);
          if (occupancy == 0) {
            break;
          }
        }
      }

      assert(offset == (piece == kBishop ? bishopAttackReachableTable.size() : rookAttackReachableTable.size()));
    }
    return table;
  }

public:
  template <Piece piece>
  inline static Bitboard getAttack(Square square, Bitboard occupancy) {
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

///////////////////////////////////////////////////////
//                   UTILITY TABLES
///////////////////////////////////////////////////////
inline const auto kSquareBetweenTable = []() {
  std::array<std::array<Bitboard, kSquareSize>, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    for (Square j = 0; j < kSquareSize; ++j) {
      if (i == j) {
        continue;
      }
      if (kSquareToDiagonalMasks[i] == kSquareToDiagonalMasks[j] || kSquareToAntiDiagonalMaskTable[i] == kSquareToAntiDiagonalMaskTable[j]) {
        table[i][j] = SliderTable::getAttack<kBishop>(i, toBitboard(j)) & SliderTable::getAttack<kBishop>(j, toBitboard(i));
      } else if (getSquareRank(i) == getSquareRank(j) || getSquareFile(i) == getSquareFile(j)) {
        table[i][j] = SliderTable::getAttack<kRook>(i, toBitboard(j)) & SliderTable::getAttack<kRook>(j, toBitboard(i));
      }
    }
  }
  return table;
}();
