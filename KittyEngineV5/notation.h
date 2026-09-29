#pragma once
#include "bitboard.h"
#include <string>
#include <string_view>
#include <utility>

namespace bb::notation {
  char pieceToAsciiVisualOnly(Color color, Piece piece);
  char pieceToAscii(Color color, Piece piece);
  std::pair<Color, Piece> asciiToPiece(char ascii);
  Color parseSideToMove(std::string_view side);
  Bitboard parseCastlingRights(std::string_view rights);
  std::string colorToString(Color color);
  std::string squareToString(Square square);
  Square stringToSquare(std::string_view square);
  std::string castleToString(Bitboard permission);
}
