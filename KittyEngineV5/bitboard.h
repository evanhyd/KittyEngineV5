#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <format>
#include <iostream>
#include <stdint.h>
#include <string>
#include <utility>

///////////////////////////////////////////////////////
//                 BITBOARD DEFINITION
///////////////////////////////////////////////////////
using Bitboard = uint64_t;
using Square = uint32_t;
using Color = int32_t;
using Piece = int32_t;

enum : Square {
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

enum : Color {
  kWhite,
  kBlack,
};

enum : Piece {
  kPawn,
  kKnight,
  kBishop,
  kRook,
  kQueen,
  kKing,
  kNoPiece,
};


///////////////////////////////////////////////////////
//                 GEOMETRY DEFINITION
///////////////////////////////////////////////////////
inline constexpr Bitboard kFileAMask = 0x101010101010101ull;
inline constexpr Bitboard kFileBMask = kFileAMask << 1;
inline constexpr Bitboard kFileCMask = kFileAMask << 2;
inline constexpr Bitboard kFileDMask = kFileAMask << 3;
inline constexpr Bitboard kFileEMask = kFileAMask << 4;
inline constexpr Bitboard kFileFMask = kFileAMask << 5;
inline constexpr Bitboard kFileGMask = kFileAMask << 6;
inline constexpr Bitboard kFileHMask = kFileAMask << 7;
inline constexpr Bitboard kRank8Mask = 0xffull;
inline constexpr Bitboard kRank7Mask = kRank8Mask << 8;
inline constexpr Bitboard kRank6Mask = kRank8Mask << 16;
inline constexpr Bitboard kRank5Mask = kRank8Mask << 24;
inline constexpr Bitboard kRank4Mask = kRank8Mask << 32;
inline constexpr Bitboard kRank3Mask = kRank8Mask << 40;
inline constexpr Bitboard kRank2Mask = kRank8Mask << 48;
inline constexpr Bitboard kRank1Mask = kRank8Mask << 56;


///////////////////////////////////////////////////////
//                 HELPER FUNCTION
///////////////////////////////////////////////////////
// Funny optimization: https://godbolt.org/z/GnbKzd33s
///////////////////////////////////////////////////////
template <typename ...Squares>
[[nodiscard]] inline constexpr Bitboard toBitboard(Squares... squares) { return ((1ull << squares) | ...); }
[[nodiscard]] inline constexpr bool isSquareSet(Bitboard bitboard, Square square) { return bitboard >> square & 1; }
[[nodiscard]] inline constexpr Bitboard setSquare(Bitboard bitboard, Square square) { return bitboard | 1ull << square; }
[[nodiscard]] inline constexpr Bitboard unsetSquare(Bitboard bitboard, Square square) { return bitboard & ~(1ull << square); }
[[nodiscard]] inline constexpr Bitboard moveSquare(Bitboard bitboard, Square from, Square to) { return bitboard & ~(1ull << from) | (1ull << to); }
[[nodiscard]] inline constexpr Bitboard shiftUp(Bitboard bitboard) { return bitboard >> 8; }
[[nodiscard]] inline constexpr Bitboard shiftDown(Bitboard bitboard) { return bitboard << 8; }
[[nodiscard]] inline constexpr Bitboard shiftLeft(Bitboard bitboard) { return (bitboard >> 1) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftRight(Bitboard bitboard) { return (bitboard << 1) & ~kFileAMask; }
[[nodiscard]] inline constexpr Bitboard shiftUpLeft(Bitboard bitboard) { return (bitboard >> 9) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftUpRight(Bitboard bitboard) { return (bitboard >> 7) & ~kFileAMask; }
[[nodiscard]] inline constexpr Bitboard shiftDownLeft(Bitboard bitboard) { return (bitboard << 7) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftDownRight(Bitboard bitboard) { return (bitboard << 9) & ~kFileAMask; }

[[nodiscard]] inline constexpr uint32_t countPiece(Bitboard bitboard) { return static_cast<uint32_t>(std::popcount(bitboard)); }
[[nodiscard]] inline constexpr Square peekPiece(Bitboard bitboard) { return static_cast<Square>(std::countr_zero(bitboard)); }
[[nodiscard]] inline constexpr Bitboard popPiece(Bitboard bitboard) { return bitboard & (bitboard - 1); }
[[nodiscard]] inline constexpr Square getSquareRank(Square square) { return square / 8; }
[[nodiscard]] inline constexpr Square getSquareFile(Square square) { return square % 8; }
[[nodiscard]] inline constexpr Square rankFileToSquare(Square rank, Square file) { return rank * 8 + file; }
[[nodiscard]] inline constexpr Square squareUp(Square square) { return square - 8; }
[[nodiscard]] inline constexpr Square squareDown(Square square) { return square + 8; }
[[nodiscard]] inline constexpr Square squareUpLeft(Square square) { return square - 9; }
[[nodiscard]] inline constexpr Square squareUpRight(Square square) { return square - 7; }
[[nodiscard]] inline constexpr Square squareDownLeft(Square square) { return square + 7; }
[[nodiscard]] inline constexpr Square squareDownRight(Square square) { return square + 9; }

[[nodiscard]] inline constexpr Color getOtherColor(Color color) { return (color == kWhite ? kBlack : kWhite); }


///////////////////////////////////////////////////////
//                 GAME DEFINITION
///////////////////////////////////////////////////////
inline constexpr Square kColorSize = 2;
inline constexpr Square kPieceSize = 6;
inline constexpr Square kSideSize = 8;
inline constexpr Square kSquareSize = 64;
inline constexpr std::array<Bitboard, kColorSize> kBackRank = {getSquareRank(A1), getSquareRank(A8) };
inline constexpr std::array<Bitboard, kColorSize> kPromotionRank = { getSquareRank(A8), getSquareRank(A1) };
inline constexpr std::array<Bitboard, kColorSize> kKingCastlePermission = { toBitboard(E1, H1) , toBitboard(E8, H8) };
inline constexpr std::array<Bitboard, kColorSize> kQueenCastlePermission = { toBitboard(E1, A1) , toBitboard(E8, A8) };
inline constexpr std::array<Bitboard, kColorSize> kKingCastleOccupancy = { toBitboard(F1, G1) , toBitboard(F8, G8) };
inline constexpr std::array<Bitboard, kColorSize> kQueenCastleOccupancy = { toBitboard(B1, C1, D1) , toBitboard(B8, C8, D8) };
inline constexpr std::array<Bitboard, kColorSize> kKingCastleSafety = { toBitboard(E1, F1, G1) , toBitboard(E8, F8, G8) };
inline constexpr std::array<Bitboard, kColorSize> kQueenCastleSafety = { toBitboard(C1, D1, E1) , toBitboard(C8, D8, E8) };


///////////////////////////////////////////////////////
//                 MASK TABLES
///////////////////////////////////////////////////////
inline constexpr auto kSquareToRankMasks = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = kRank8Mask << (getSquareRank(i) * kSideSize);
  }
  return table;
}();

inline constexpr auto kSquareToFileMasks = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = kFileAMask << getSquareFile(i);
  }
  return table;
}();

inline constexpr auto kSquareToDiagonalMasks = []() {
  auto kDiagonalMasks = []() {
    std::array<Bitboard, 15> table{ 1 };
    for (Square i = 1; i < table.size(); ++i) {
      table[i] = shiftRight(table[i - 1]) | shiftDown(table[i - 1]);
    }
    return table;
  }();

  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = kDiagonalMasks[getSquareRank(i) + getSquareFile(i)];
  }
  return table;
}();

inline constexpr auto kSquareToAntiDiagonalMaskTable = []() {
  auto kAntiDiagonalMasks = []() {
    std::array<Bitboard, 15> table{ 0x80 };
    for (Square i = 1; i < table.size(); ++i) {
      table[i] = shiftLeft(table[i - 1]) | shiftDown(table[i - 1]);
    }
    return table;
  }();

  std::array<Bitboard, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    table[i] = kAntiDiagonalMasks[getSquareRank(i) + kSideSize - getSquareFile(i) - 1];
  }
  return table;
}();


///////////////////////////////////////////////////////
//                   ATTACK TABLES
///////////////////////////////////////////////////////
namespace internal {
  inline constexpr auto kPawnAttackTable = []() {
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

  inline constexpr auto kKnightAttackTable = []() {
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

  inline constexpr auto kKingAttackTable = []() {
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

  // A plain magic bitboard implementation.
  // Avoid using PEXT on AMD due to slow implementation.
  struct SliderAttackTable {
    Bitboard magicNum;          // Magic bitboard hashing
    Bitboard maxAttackNoEdge;   // Maximum attack pattern excludes the border
    Bitboard* attackReachable;  // Reachable attack table range, indexed by magic shifting

    template <Piece piece>
    constexpr size_t getKey(Bitboard occupancy) const {
      constexpr size_t relevantBitsInverse = (piece == kBishop ? kSquareSize - 9 : kSquareSize - 12);
      return ((occupancy & maxAttackNoEdge) * magicNum) >> relevantBitsInverse;
    }

    template <Piece piece>
    constexpr Bitboard getAttack(Bitboard occupancy) const {
      return attackReachable[getKey<piece>(occupancy)];
    }
  };

  // Generate attack ray bitboard at square, spans outward and stops at occupancy bits at each direction.
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

  inline std::array<Bitboard, 64 * 512> bishopAttackReachableTable{};
  inline std::array<Bitboard, 64 * 4096> rookAttackReachableTable{};

  inline const auto sliderAttackTables = []() {
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
    } };

    std::array<std::array<SliderAttackTable, kColorSize>, kSquareSize> table{};

    // Generate for both bishop and rook.
    for (Piece piece : {kBishop, kRook}) {
      size_t offset = 0;

      for (Square i = 0; i < kSquareSize; ++i) {
        SliderAttackTable& magic = table[i][piece - kBishop];

        // Set up magic bitboard hashing factors.
        magic.magicNum = kMagicNumTable[piece - kBishop][i];

        // Remove the edge since the sliding piece must stop at the edge. This reduces the occupancy permutation size.
        Bitboard edge = ((kRank1Mask | kRank8Mask) & ~kSquareToRankMasks[i]) | ((kFileAMask | kFileHMask) & ~kSquareToFileMasks[i]);
        magic.maxAttackNoEdge = internal::generateSliderAttackReachable(piece, i, 0) & ~edge;

        // Assign the attack table range.
        magic.attackReachable = (piece == kBishop ? bishopAttackReachableTable.data() : rookAttackReachableTable.data()) + offset;

        size_t permutations = (piece == kBishop ? (1ull << 9) : (1ull << 12));
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
  }();
}

template <Piece piece, Color color>
  requires (piece == kPawn)
inline constexpr Bitboard getAttack(Square square) {
  return internal::kPawnAttackTable[color][square];
}

template <Piece piece>
  requires (piece == kKnight || piece == kKing)
inline constexpr Bitboard getAttack(Square square) {
  if constexpr (piece == kKnight) {
    return internal::kKnightAttackTable[square];
  } else if constexpr (piece == kKing) {
    return internal::kKingAttackTable[square];
  }
}

template <Piece piece>
requires (piece == kBishop || piece == kRook || piece == kQueen)
inline constexpr Bitboard getAttack(Square square, Bitboard occupancy) {
  if constexpr (piece == kBishop) {
    return internal::sliderAttackTables[square][0].getAttack<kBishop>(occupancy);
  } else if constexpr (piece == kRook) {
    return internal::sliderAttackTables[square][1].getAttack<kRook>(occupancy);
  } else if constexpr (piece == kQueen) {
    return internal::sliderAttackTables[square][0].getAttack<kBishop>(occupancy) |
           internal::sliderAttackTables[square][1].getAttack<kRook>(occupancy);
  }
}


///////////////////////////////////////////////////////
//                   UTILITY TABLES
///////////////////////////////////////////////////////
inline constexpr auto kLineOfSightMasks = []() {
  std::array<std::array<Bitboard, kSquareSize>, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    for (Square j = 0; j < kSquareSize; ++j) {
      if (i == j) {
        continue;
      }

      if (kSquareToRankMasks[i] == kSquareToRankMasks[j]) {
        table[i][j] = kSquareToRankMasks[i];
      } else if (kSquareToFileMasks[i] == kSquareToFileMasks[j]) {
        table[i][j] = kSquareToFileMasks[i];
      } else if (kSquareToDiagonalMasks[i] == kSquareToDiagonalMasks[j]) {
        table[i][j] = kSquareToDiagonalMasks[i];
      } else if (kSquareToAntiDiagonalMaskTable[i] == kSquareToAntiDiagonalMaskTable[j]) {
        table[i][j] = kSquareToAntiDiagonalMaskTable[i];
      }
    }
  }
  return table;
}();

inline const auto kSquareBetweenMasks = []() {
  std::array<std::array<Bitboard, kSquareSize>, kSquareSize> table{};
  for (Square i = 0; i < kSquareSize; ++i) {
    for (Square j = 0; j < kSquareSize; ++j) {
      if (i == j) {
        continue;
      }
      if (kSquareToDiagonalMasks[i] == kSquareToDiagonalMasks[j] || kSquareToAntiDiagonalMaskTable[i] == kSquareToAntiDiagonalMaskTable[j]) {
        table[i][j] = getAttack<kBishop>(i, toBitboard(j)) & getAttack<kBishop>(j, toBitboard(i));
      } else if (getSquareRank(i) == getSquareRank(j) || getSquareFile(i) == getSquareFile(j)) {
        table[i][j] = getAttack<kRook>(i, toBitboard(j)) & getAttack<kRook>(j, toBitboard(i));
      }
    }
  }
  return table;
}();


///////////////////////////////////////////////////////
//                 UI FORMATTING
///////////////////////////////////////////////////////
inline void printBitboard(Bitboard bitboard) {
  using std::cout;
  using std::format;
  for (Square i = 0; i < kSideSize; ++i) {
    cout << format("{}|", kSideSize - i);
    for (Square j = 0; j < kSideSize; ++j) {
      cout << format(" {:d}", isSquareSet(bitboard, rankFileToSquare(i, j)));
    }
    cout << '\n';
  }
  cout << format("   a b c d e f g h\nBitboard Hex: {:#018x}\n\n", bitboard);
}

inline char pieceToAsciiVisualOnly(Color color, Piece piece) {
  constexpr std::array<std::array<char, kPieceSize>, kColorSize> table = { {
    {'A', 'N', 'B', 'R', 'Q', 'K'},
    {'v', 'n', 'b', 'r', 'q', 'k'},
  } };
  return table[color][piece];
}

inline char pieceToAscii(Color color, Piece piece) {
  constexpr std::array<std::array<char, kPieceSize>, kColorSize> table = { {
    {'P', 'N', 'B', 'R', 'Q', 'K'},
    {'p', 'n', 'b', 'r', 'q', 'k'},
  } };
  return table[color][piece];
}

inline std::pair<Color, Piece> asciiToPiece(char ascii) {
  for (Color color : {kWhite, kBlack}) {
    for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
      if (pieceToAscii(color, piece) == ascii) {
        return { color, piece };
      }
    }
  }
  assert(false && "invalid ascii value");
  return {}; // unreachable
}

inline std::string colorToString(Color color) {
  static const std::array<std::string, kColorSize> table = {
    "white",
    "black",
  };
  return table[color];
}

inline std::string squareToString(Square square) {
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

inline Square stringToSquare(const std::string& squareString) {
  for (Square i = 0; i < kSquareSize; ++i) {
    if (squareToString(i) == squareString) {
      return i;
    }
  }
  assert(false && "invalid square");
  return NO_SQUARE;
}

inline std::string castleToString(Bitboard permission) {
  std::string str;
  if (permission & kKingCastlePermission[kWhite]) {
    str += 'K';
  }
  if (permission & kQueenCastlePermission[kWhite]) {
    str += 'Q';
  }
  if (permission & kKingCastlePermission[kBlack]) {
    str += 'k';
  }
  if (permission & kQueenCastlePermission[kBlack]) {
    str += 'q';
  }
  if (str.empty()) {
    str = "-";
  }
  return str;
}
