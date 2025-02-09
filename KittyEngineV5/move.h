#pragma once
#include "bitboard.h"
#include <format>
#include <iostream>
#include <type_traits>

///////////////////////////////////////////////////////
//            MOVE DEFINITION
//
//            binary move bits
//     0000 0000 0000 0000 0000 0000 0011 1111    source square
//     0000 0000 0000 0000 0000 1111 1100 0000    destination square
//     0000 0000 0000 0000 1111 0000 0000 0000    moved piece
//     0000 0000 0000 1111 0000 0000 0000 0000    promoted piece
//     0000 0000 0001 0000 0000 0000 0000 0000    capture flag
//     0000 0000 0010 0000 0000 0000 0000 0000    enpassant flag
//     0000 0000 0100 0000 0000 0000 0000 0000    double push flag
//     0000 0000 1000 0000 0000 0000 0000 0000    castling flag
// 
/////////////////////////////////////////////////////////
class Move {
  uint32_t move_;

public:
  static constexpr uint32_t kCaptureFlag = 0x100000ull;
  static constexpr uint32_t kEnpassantFlag = 0x200000ull;
  static constexpr uint32_t kDoublePushFlag = 0x400000ull;
  static constexpr uint32_t kCastlingFlag = 0x800000ull;

  Move() = default;

  explicit constexpr Move(uint32_t sourceSquare, uint32_t destinationSquare, uint32_t movedPiece, uint32_t promotedPiece = 0, uint32_t flag = 0) 
    : move_(sourceSquare | destinationSquare << 6 | movedPiece << 12 | promotedPiece << 16 | flag) {
  }

  constexpr uint32_t getSourceSquare() const { return move_ & 0b111111; }
  constexpr uint32_t getDestinationSquare() const { return move_ >> 6 & 0b111111; }
  constexpr uint32_t getMovedPiece() const { return move_ >> 12 & 0b1111; }
  constexpr uint32_t getPromotedPiece() const { return move_ >> 16 & 0b1111; }
  constexpr bool isCaptured() const { return move_ & kCaptureFlag; }
  constexpr bool isEnpassant() const { return move_ & kEnpassantFlag; }
  constexpr bool isDoublePush() const { return move_ & kDoublePushFlag; }
  constexpr bool isCastling() const { return move_ & kCastlingFlag; }

  std::string toString() const {
    uint32_t promotedPiece = getPromotedPiece();
    return std::format("{}{}{}", 
              squareToString(getSourceSquare()), 
              squareToString(getDestinationSquare()), 
              (promotedPiece == 0 ? ' ' : pieceToAscii(kBlack, static_cast<Piece>(promotedPiece))));
  }
};
static_assert(std::is_trivial_v<Move>, "Move is not POD type, may affect performance");

class MoveList {
  std::array<Move, 256> moves_;
  size_t size_;

public:
  void push(Move move) {
    moves_[size_] = move;
    ++size_;
  }

  inline friend std::ostream& operator<<(std::ostream& out, const MoveList& moveList) {
    out << std::format("Total Moves: {}\n", moveList.size_);
    for (size_t i = 0; i < moveList.size_; ++i) {
      out << std::format("{}\n", moveList.moves_[i].toString());
    }
    return out;
  }
};
