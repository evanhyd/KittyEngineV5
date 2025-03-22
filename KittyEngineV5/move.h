#pragma once
#include "bitboard.h"

struct MoveType {
  Piece movedPiece;
  Piece promotionPiece;
  bool isEnpassant;
  bool isDoublePush;
  bool isKingSideCastle;
  bool isQueenSideCastle;
};

struct Move {
  Square srce;
  Square dest;
};
