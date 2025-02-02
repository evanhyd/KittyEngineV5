#pragma once
#include <iostream>
#include <array>
#include "bitboard.h"

class Board {
  std::array<std::array<Bitboard, kPieceSize>, kTeamSize> bitboards_;
  std::array<Bitboard, kTeamSize> occupancy_;
  Bitboard totalOccupancy_;
  Team team_;
  CastlePermission castlePermission_;
  Square enpassant_;
  uint32_t halfmove_;
  uint32_t fullmove_;

public:
  static Board fromFEN(const std::string& fen);
  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
