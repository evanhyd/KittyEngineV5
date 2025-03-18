#pragma once
#include "bitboard.h"
#include <type_traits>

struct MoveType {
  Color color;
  Piece movedPiece;
  Piece promotionPiece;
  bool isEnpassant;
  bool isDoublePush;
  bool isKingSideCastle;
  bool isQueenSideCastle;
};

template <MoveType moveType>
struct Move {
  Square srce;
  Square dest;
};
