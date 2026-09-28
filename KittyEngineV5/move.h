#pragma once
#include "bitboard.h"
#include <cstdint>

// MOVE ENCODING //
/*
      Binary move bit layout (uint32_t):

      0000 0000 0000 0000 0011 1111   source square          0x3F
      0000 0000 0000 1111 1100 0000   dest square     >>  6  & 0x3F
      0000 0000 1111 0000 0000 0000   moved piece     >> 12  & 0x0F
      0000 1111 0000 0000 0000 0000   promoted piece  >> 16  & 0x0F
      0001 0000 0000 0000 0000 0000   capture flag           0x100000
      0010 0000 0000 0000 0000 0000   enpassant flag         0x200000
      0100 0000 0000 0000 0000 0000   double push            0x400000
      1000 0000 0000 0000 0000 0000   castling flag          0x800000
  */
class Move {
  uint32_t rawMove;

  // Bitmasks
  static constexpr uint32_t kGetSquareMask = 0x3Fu;
  static constexpr uint32_t kGetPieceMask = 0x0Fu;

public:
  // Flags
  static constexpr uint32_t kCaptureFlag = 1u << 20;
  static constexpr uint32_t kEnpassantFlag = 1u << 21;
  static constexpr uint32_t kDoublePushFlag = 1u << 22;
  static constexpr uint32_t kCastlingFlag = 1u << 23;
  static constexpr uint32_t kViolentMoveFlag = (kGetPieceMask << 16) | kCaptureFlag | kEnpassantFlag;

  // Constructors
  constexpr Move() noexcept : rawMove(0) {}

  constexpr Move(Square sourceSquare, Square destSquare, Piece movedPiece,
                 Piece promotedPiece = kNoPiece, uint32_t flag = 0) noexcept
    : rawMove((sourceSquare & kGetSquareMask) |
              ((destSquare & kGetSquareMask) << 6) |
              ((static_cast<uint32_t>(movedPiece) & kGetPieceMask) << 12) |
              ((static_cast<uint32_t>(promotedPiece) & kGetPieceMask) << 16) |
              flag) {}

  // Getters
  constexpr Square getSource() const noexcept {
    return rawMove & kGetSquareMask;
  }

  constexpr Square getDest() const noexcept {
    return (rawMove >> 6) & kGetSquareMask;
  }

  constexpr Piece getMovedPiece() const noexcept {
    return static_cast<Piece>((rawMove >> 12) & kGetPieceMask);
  }

  constexpr Piece getPromotedPieceType() const noexcept {
    return static_cast<Piece>((rawMove >> 16) & kGetPieceMask);
  }

  // Predicates
  constexpr bool isCapture() const noexcept {
    return rawMove & kCaptureFlag;
  }

  constexpr bool isEnpassant() const noexcept {
    return rawMove & kEnpassantFlag;
  }

  constexpr bool isDoublePush() const noexcept {
    return rawMove & kDoublePushFlag;
  }

  constexpr bool isCastling() const noexcept {
    return rawMove & kCastlingFlag;
  }

  constexpr bool isQuietMove() const noexcept {
    return !(rawMove & kViolentMoveFlag);
  }
};
