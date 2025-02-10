#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <string>

///////////////////////////////////////////////////////
//                 BITBOARD DEFINITION
///////////////////////////////////////////////////////
using Bitboard = uint64_t;

enum : uint32_t {
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

inline constexpr uint32_t kWhiteKingCastlePermission = 0b0001ull;
inline constexpr uint32_t kWhiteQueenCastlePermission = 0b0010ull;
inline constexpr uint32_t kBlackKingCastlePermission = 0b0100ull;
inline constexpr uint32_t kBlackQueenCastlePermission = 0b1000ull;

inline constexpr uint32_t kTeamSize = 2;
inline constexpr uint32_t kPieceSize = 6;
inline constexpr uint32_t kSideSize = 8;
inline constexpr uint32_t kSquareSize = 64;
inline constexpr uint32_t kDiagonalSize = 15;
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

// Funny optimization: https://godbolt.org/z/GnbKzd33s
// Fine tuned: https://godbolt.org/z/M9r1xb6vb
[[nodiscard]] inline constexpr Bitboard setSquare(Bitboard bitboard, uint32_t square) { return bitboard | 1ull << square; }
[[nodiscard]] inline constexpr Bitboard unsetSquare(Bitboard bitboard, uint32_t square) { return bitboard & ~(1ull << square); }
[[nodiscard]] inline constexpr Bitboard moveSquare(Bitboard bitboard, uint32_t from, uint32_t to) { return bitboard & ~(1ull << from) | (1ull << to); }
[[nodiscard]] inline constexpr int countPiece(Bitboard bitboard) { return std::popcount(bitboard); }
[[nodiscard]] inline constexpr uint32_t findFirstPiece(Bitboard bitboard) { return static_cast<uint32_t>(std::countr_zero(bitboard)); }
[[nodiscard]] inline constexpr Bitboard removeFirstPiece(Bitboard bitboard) { return bitboard & (bitboard - 1); }
[[nodiscard]] inline constexpr bool isSquareSet(Bitboard bitboard, uint32_t square) { return bitboard >> square & 1; }
[[nodiscard]] inline constexpr uint32_t getSquareRank(uint32_t square) { return square / 8; }
[[nodiscard]] inline constexpr uint32_t getSquareFile(uint32_t square) { return square % 8; }
[[nodiscard]] inline constexpr uint32_t rankFileToSquare(uint32_t rank, uint32_t file) { return rank * 8 + file; }
[[nodiscard]] inline constexpr Bitboard shiftUp(Bitboard bitboard) { return bitboard >> 8; }
[[nodiscard]] inline constexpr Bitboard shiftDown(Bitboard bitboard) { return bitboard << 8; }
[[nodiscard]] inline constexpr Bitboard shiftLeft(Bitboard bitboard) { return bitboard >> 1; }
[[nodiscard]] inline constexpr Bitboard shiftRight(Bitboard bitboard) { return bitboard << 1; }
[[nodiscard]] inline constexpr Bitboard shiftUpLeft(Bitboard bitboard) { return bitboard >> 9; }
[[nodiscard]] inline constexpr Bitboard shiftUpRight(Bitboard bitboard) { return bitboard >> 7; }
[[nodiscard]] inline constexpr Bitboard shiftDownLeft(Bitboard bitboard) { return bitboard << 7; }
[[nodiscard]] inline constexpr Bitboard shiftDownRight(Bitboard bitboard) { return bitboard << 9; }

inline constexpr auto kDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 1 };
  for (uint32_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (shiftRight(table[i - 1]) & ~kFileAMask) | shiftDown(table[i - 1]);
  }
  return table;
}();

inline constexpr auto kAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kDiagonalSize> table{ 0x80 };
  for (uint32_t i = 1; i < kDiagonalSize; ++i) {
    table[i] = (shiftLeft(table[i - 1]) & ~kFileHMask) | shiftDown(table[i - 1]);
  }
  return table;
}();

inline constexpr auto kSquareToRankMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kRank8Mask << (getSquareRank(i) * kSideSize);
  }
  return table;
}();

inline constexpr auto kSquareToFileMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kFileAMask << getSquareFile(i);
  }
  return table;
}();

inline constexpr auto kSquareToDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kDiagonalMaskTable[getSquareRank(i) + getSquareFile(i)];
  }
  return table;
}();

inline constexpr auto kSquareToAntiDiagonalMaskTable = []() {
  std::array<Bitboard, kSquareSize> table{};
  for (uint32_t i = 0; i < kSquareSize; ++i) {
    table[i] = kAntiDiagonalMaskTable[getSquareRank(i) + kSideSize - getSquareFile(i) - 1];
  }
  return table;
}();


///////////////////////////////////////////////////////
//                 UI FORMATTING
///////////////////////////////////////////////////////
char pieceToAsciiVisualOnly(Team team, Piece piece);
char pieceToAscii(Team team, Piece piece);
std::pair<Team, Piece> asciiToPiece(char ascii);
std::string teamToString(Team team);
std::string squareToString(uint32_t square);
uint32_t stringToSquare(const std::string& squareString);
std::string castleToString(uint32_t permission);
void printBitboard(Bitboard bitboard);
