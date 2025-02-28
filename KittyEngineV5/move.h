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
// 
/////////////////////////////////////////////////////////
class Move {
  uint32_t move_;

public:
  static constexpr uint32_t kCaptureFlag = 0x100000ull;

  Move() = default;

  explicit constexpr Move(Square source, Square destination, Piece movedPiece)
    : move_(static_cast<uint32_t>(source) | static_cast<uint32_t>(destination) << 6 | static_cast<uint32_t>(movedPiece) << 12) {
  }

  explicit constexpr Move(Square source, Square destination, Piece movedPiece, Piece promotedPiece, uint32_t flag)
    : move_(static_cast<uint32_t>(source) | static_cast<uint32_t>(destination) << 6 | static_cast<uint32_t>(movedPiece) << 12 | static_cast<uint32_t>(promotedPiece) << 16 | flag) {
  }

  bool operator<(const Move& move) const { return move_ < move.move_; }
  constexpr Square getSourceSquare() const { return move_ & 0b111111; }
  constexpr Square getDestinationSquare() const { return move_ >> 6 & 0b111111; }
  constexpr Piece getMovedPiece() const { return move_ >> 12 & 0b1111; }
  constexpr Piece getPromotedPiece() const { return move_ >> 16 & 0b1111; }
  constexpr bool isCaptured() const { return move_ & kCaptureFlag; }

  std::string toString() const {
    Piece promotedPiece = getPromotedPiece();
    return std::format("{}{}{}",
                       squareToString(getSourceSquare()),
                       squareToString(getDestinationSquare()),
                       (promotedPiece == 0 ? ' ' : pieceToAscii(kBlack, promotedPiece)));
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

  constexpr Move operator[](size_t i) const { return moves_[i]; }
  constexpr size_t size() const { return size_; }
  constexpr auto begin() const { return moves_.begin(); }
  constexpr auto end() const { return moves_.begin() + size_; }

  inline friend std::ostream& operator<<(std::ostream& out, const MoveList& moveList) {
    out << std::format("Total Moves: {}\n", moveList.size());
    for (Move move : moveList) {
      out << std::format("{}\n", move.toString());
    }
    return out;
  }
};
