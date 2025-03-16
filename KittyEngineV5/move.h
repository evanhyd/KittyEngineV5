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
//     0000 1111 1111 1111 0000 0000 0000 0000    flag
//     1111 0000 0000 0000 0000 0000 0000 0000    promoted piece
// 
/////////////////////////////////////////////////////////
class Move {
  uint32_t move_;

public:
  static constexpr uint32_t kEnpassantFlag = 1 << 16;
  static constexpr uint32_t kDoublePushFlag = 1 << 17;
  static constexpr uint32_t kKingSideCastleFlag = 1 << 18;
  static constexpr uint32_t kQueenSideCastleFlag = 1 << 19;

  Move() = default;

  explicit constexpr Move(Square source, Square destination, Piece movedPiece, Piece promotedPiece, uint32_t flags)
    : Move(source, destination, movedPiece, promotedPiece) {
    move_ |= flags;
  }

  explicit constexpr Move(Square source, Square destination, Piece movedPiece, Piece promotedPiece)
    : Move(source, destination, movedPiece) {
    move_ |= static_cast<uint32_t>(promotedPiece) << 28;
  }

  explicit constexpr Move(Square source, Square destination, Piece movedPiece)
    : move_(static_cast<uint32_t>(source) |
            static_cast<uint32_t>(destination) << 6 |
            static_cast<uint32_t>(movedPiece) << 12) {
  }

  constexpr Square getSourceSquare() const { return move_ & 0b111111; }
  constexpr Square getDestinationSquare() const { return move_ >> 6 & 0b111111; }
  constexpr Piece getMovedPiece() const { return move_ >> 12 & 0b1111; }
  constexpr Piece getPromotedPiece() const { return move_ >> 28 & 0b1111; }
  constexpr bool isEnpassant() const { return move_ & kEnpassantFlag; }
  constexpr bool isDoublePush() const { return move_ & kDoublePushFlag; }
  constexpr bool isKingSideCastle() const { return move_ & kKingSideCastleFlag; }
  constexpr bool isQueenSideCastle() const { return move_ & kQueenSideCastleFlag; }

  std::string toString() const {
    std::string str = std::format("{}{}", squareToString(getSourceSquare()), squareToString(getDestinationSquare()));
    if (Piece promotedPiece = getPromotedPiece(); promotedPiece) {
      str += pieceToAscii(kBlack, promotedPiece);
    }
    return str;
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
