#pragma once
#include "bitboard.h"
#include <string>
#include <string_view>

namespace bb::user_interface {
  class UserInterface {

  };

  // Print a bitboard to the console in a human-readable format.
  void printBitboard(bb::Bitboard bitboard);

  // Piece to ascii conversion for visual only, e.g. (0, 0) -> 'A', (1, 5) -> 'k'
  char pieceToAsciiVisualOnly(bb::Color color, bb::Piece piece);

  // Piece to ascii conversion, e.g. (0, 0) -> 'P', (1, 5) -> 'k'
  char pieceToAscii(bb::Color color, bb::Piece piece);

  // Ascii to piece conversion, e.g. 'P' -> (0, 0), 'k' -> (1, 5)
  std::pair<bb::Color, bb::Piece> asciiToPiece(char ascii);

  // Color to string conversion, e.g. 0 -> "white", 1 -> "black"
  std::string colorToString(bb::Color color);

  // Square number to string conversion, e.g. 0 -> "a8", 63 -> "h1"
  std::string squareToString(bb::Square square);

  // String to square number conversion, e.g. "a8" -> 0, "h1" -> 63
  bb::Square stringToSquare(std::string_view squareString);

  // Castle permission bitboard to string conversion, e.g. 0b0001 -> "K", 0b0010 -> "Q", 0b0100 -> "k", 0b1000 -> "q"
  std::string castleToString(bb::Bitboard permission);
}
