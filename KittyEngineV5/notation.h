#pragma once
#include "bitboard.h"
#include "move.h"
#include <string>
#include <string_view>
#include <utility>

namespace bb {
  class BoardState;
}

namespace bb::notation {
  // Piece ascii.
  char pieceToAsciiVisual(Color color, Piece piece);
  char pieceToAscii(Color color, Piece piece);
  std::pair<Color, Piece> asciiToPiece(char ascii);

  // Side to play.
  Color stringToSide(std::string_view side);
  std::string colorToString(Color color);

  // Square.
  Square stringToSquare(std::string_view square);
  std::string squareToString(Square square);

  // Castling.
  Bitboard stringToCastling(std::string_view rights);
  std::string castleToString(Bitboard permission);

  // Full six-field FEN for a position.
  std::string boardToFen(const BoardState& state);

  // Move.
  //Move stringToMove(std::string_view moveNotation);
  std::string moveToString(Move move);
}
