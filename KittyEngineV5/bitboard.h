#pragma once
#include <string>
#include <cstdint>
#include <cassert>

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

constexpr uint32_t kTeamSize = 2;
constexpr uint32_t kPieceSize = 6;
constexpr uint32_t kSideSize = 8;
constexpr uint32_t kSquareSize = kSideSize * kSideSize;

template <typename ...Squares>
[[nodiscard]] constexpr Bitboard setSquare(Bitboard bitboard, Squares... squares) {
  (assert(squares < kSquareSize), ...);
  return bitboard | ((1ull << squares) | ...);
}

constexpr bool isSquareSet(Bitboard bitboard, uint32_t square) {
  assert(square < kSquareSize);
  return bitboard >> square & 1;
}

char pieceToAscii(Team team, Piece piece);
std::pair<Team, Piece> asciiToPiece(char ascii);
std::string teamToString(Team team);
std::string squareToString(Square square);
Square squareStringToSquare(const std::string& squareString);
std::string castleToString(CastlePermission permission);
void printBitboard(uint64_t bitboard);
