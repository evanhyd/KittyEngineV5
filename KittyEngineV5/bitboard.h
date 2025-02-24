#pragma once
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>

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
[[nodiscard]] inline constexpr Bitboard setSquare(Bitboard bitboard, Square square) { return bitboard | 1ull << square; }
[[nodiscard]] inline constexpr Bitboard unsetSquare(Bitboard bitboard, Square square) { return bitboard & ~(1ull << square); }
[[nodiscard]] inline constexpr Bitboard moveSquare(Bitboard bitboard, Square from, Square to) { return bitboard & ~(1ull << from) | (1ull << to); }
[[nodiscard]] inline constexpr uint32_t countPiece(Bitboard bitboard) { return static_cast<uint32_t>(std::popcount(bitboard)); }
[[nodiscard]] inline constexpr Square getFirstPieceSquare(Bitboard bitboard) { return static_cast<Square>(std::countr_zero(bitboard)); }
[[nodiscard]] inline constexpr Bitboard removeFirstPiece(Bitboard bitboard) { return bitboard & (bitboard - 1); }
[[nodiscard]] inline constexpr bool isSquareSet(Bitboard bitboard, Square square) { return bitboard >> square & 1; }
[[nodiscard]] inline constexpr Square getSquareRank(Square square) { return square / 8; }
[[nodiscard]] inline constexpr Square getSquareFile(Square square) { return square % 8; }
[[nodiscard]] inline constexpr Square rankFileToSquare(Square rank, Square file) { return rank * 8 + file; }
[[nodiscard]] inline constexpr Bitboard shiftUp(Bitboard bitboard) { return bitboard >> 8; }
[[nodiscard]] inline constexpr Bitboard shiftDown(Bitboard bitboard) { return bitboard << 8; }
[[nodiscard]] inline constexpr Bitboard shiftLeft(Bitboard bitboard) { return (bitboard >> 1) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftRight(Bitboard bitboard) { return (bitboard << 1) & ~kFileAMask; }
[[nodiscard]] inline constexpr Bitboard shiftUpLeft(Bitboard bitboard) { return (bitboard >> 9) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftUpRight(Bitboard bitboard) { return (bitboard >> 7) & ~kFileAMask; }
[[nodiscard]] inline constexpr Bitboard shiftDownLeft(Bitboard bitboard) { return (bitboard << 7) & ~kFileHMask; }
[[nodiscard]] inline constexpr Bitboard shiftDownRight(Bitboard bitboard) { return (bitboard << 9) & ~kFileAMask; }
[[nodiscard]] inline consteval Color getOtherColor(Color color) { return (color == kWhite ? kBlack : kWhite); }


///////////////////////////////////////////////////////
//                 GAME DEFINITION
///////////////////////////////////////////////////////
inline constexpr Square kColorSize = 2;
inline constexpr Square kPieceSize = 6;
inline constexpr Square kSideSize = 8;
inline constexpr Square kSquareSize = 64;
inline constexpr std::array<Bitboard, kColorSize> kBackRank = {getSquareRank(A1), getSquareRank(A8) };
inline constexpr std::array<Bitboard, kColorSize> kPromotionRank = { getSquareRank(A8), getSquareRank(A1) };
inline constexpr Bitboard kWhiteKingCastlePermission = toBitboard(E1, H1);
inline constexpr Bitboard kWhiteQueenCastlePermission = toBitboard(E1, A1);
inline constexpr Bitboard kBlackKingCastlePermission = toBitboard(E8, H8);
inline constexpr Bitboard kBlackQueenCastlePermission = toBitboard(E8, A8);


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
  if (permission & kWhiteKingCastlePermission) {
    str += 'K';
  }
  if (permission & kWhiteQueenCastlePermission) {
    str += 'Q';
  }
  if (permission & kBlackKingCastlePermission) {
    str += 'k';
  }
  if (permission & kBlackQueenCastlePermission) {
    str += 'q';
  }
  if (str.empty()) {
    str = "-";
  }
  return str;
}
