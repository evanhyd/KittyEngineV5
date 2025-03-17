#pragma once
#include "move.h"
#include <array>

///////////////////////////////////////////////////////
//                 CHESS BOARD STATUS
///////////////////////////////////////////////////////
//struct BoardStatus {
//  Color color;
//  bool canEnpassant;
//  std::array<bool, kColorSize> kingCastle;
//  std::array<bool, kColorSize> queenCastle;
//
//  constexpr BoardStatus enpassantMove() const {
//    BoardStatus status{};
//    status.color = !color;
//    status.canEnpassant = true;
//    status.kingCastle = kingCastle;
//    status.queenCastle = queenCastle;
//    return status;
//  }
//
//  constexpr BoardStatus kingCastleMove() const {
//    BoardStatus status{};
//    status.color = !color;
//    status.canEnpassant = false;
//    status.kingCastle[color] = false;
//    status.kingCastle[!color] = kingCastle[!color];
//    status.queenCastle = queenCastle;
//    return status;
//  }
//
//  constexpr BoardStatus queenCastleMove() const {
//    BoardStatus status{};
//    status.color = !color;
//    status.canEnpassant = false;
//    status.kingCastle = kingCastle;
//    status.queenCastle[color] = false;
//    status.queenCastle[!color] = queenCastle[!color];
//    return status;
//  }
//};


///////////////////////////////////////////////////////
//                 CHESS BOARD STATE
///////////////////////////////////////////////////////
class BoardState {
public:
  std::array<std::array<Bitboard, kPieceSize>, kColorSize> bitboards_;
  Bitboard castlePermission_;
  Square enpassant_;
  uint32_t halfmove_;
  uint32_t fullmove_;
  Color color_;

  // Return a bitboard containing squares attacked by their pieces.
  template <Color our>
  constexpr Bitboard getAttackedMask(Bitboard bothOccupancy) const {
    constexpr Color their = getOtherColor(our);

    // If king blocks the attack ray, then it may incorrectly move backward illegally.
    // Consider the move: r...K... -> r....K..
    const Bitboard occupancy = bothOccupancy & ~bitboards_[our][kKing];

    // Calculate the attack masks.
    Bitboard attackedMask = (their == kWhite ?
                         shiftUpLeft(bitboards_[their][kPawn]) | shiftUpRight(bitboards_[their][kPawn]) :
                         shiftDownLeft(bitboards_[their][kPawn]) | shiftDownRight(bitboards_[their][kPawn]));

    attackedMask |= getAttack<kKing>(peekPiece(bitboards_[their][kKing]));

    for (Bitboard bb = bitboards_[their][kKnight]; bb; bb = popPiece(bb)) {
      attackedMask |= getAttack<kKnight>(peekPiece(bb));
    }
    for (Bitboard bb = bitboards_[their][kBishop]; bb; bb = popPiece(bb)) {
      attackedMask |= getAttack<kBishop>(peekPiece(bb), occupancy);
    }
    for (Bitboard bb = bitboards_[their][kRook]; bb; bb = popPiece(bb)) {
      attackedMask |= getAttack<kRook>(peekPiece(bb), occupancy);
    }
    for (Bitboard bb = bitboards_[their][kQueen]; bb; bb = popPiece(bb)) {
      attackedMask |= getAttack<kQueen>(peekPiece(bb), occupancy);
    }
    return attackedMask;
  }

  // Return a bitboard containing the intersection of all attacks.
  // Must block the attack or capture the attackers.
  template <Color our>
  constexpr Bitboard getCheckedMask(Square kingSq, Bitboard bothOccupancy) const {
    constexpr Color their = getOtherColor(our);

    Bitboard checkedMask = ~Bitboard{};
    Bitboard checkers = (getAttack<kPawn, our>(kingSq) & bitboards_[their][kPawn]) |
                        (getAttack<kKnight>(kingSq) & bitboards_[their][kKnight]) |
                        (getAttack<kBishop>(kingSq, bothOccupancy) & (bitboards_[their][kBishop] | bitboards_[their][kQueen])) |
                        (getAttack<kRook>(kingSq, bothOccupancy) & (bitboards_[their][kRook] | bitboards_[their][kQueen]));

    for (;checkers; checkers = popPiece(checkers)) {
      const Square sq = peekPiece(checkers);
      checkedMask &= setSquare(kSquareBetweenMasks[kingSq][sq], sq);
    }
    return checkedMask;
  }

  // Return a bitboard containing our pieces that are pinned.
  template <Color our>
  constexpr Bitboard getPinnedMask(Square kingSq , const std::array<Bitboard, kColorSize> occupancy) const {
    constexpr Color their = getOtherColor(our);

    // Get the enemy sliders squares, then check if any ally piece is blocking the attack ray.
    Bitboard pinned{};
    Bitboard sliders = (getAttack<kBishop>(kingSq, occupancy[their]) & (bitboards_[their][kBishop] | bitboards_[their][kQueen])) |
                       (getAttack<kRook>(kingSq, occupancy[their]) & (bitboards_[their][kRook] | bitboards_[their][kQueen]));
    for (; sliders; sliders = popPiece(sliders)) {
      Square sliderSquare = peekPiece(sliders);
      Bitboard blockers = kSquareBetweenMasks[kingSq][sliderSquare] & occupancy[our];
      if (popPiece(blockers) == 0) {
        pinned |= blockers; // Does NOT handle enpassant edge case.
      }
    }
    return pinned;
  }

  template <Color our, Piece piece, typename Receiver>
  constexpr void getPieceMove(const Square kingSq, const std::array<Bitboard, kColorSize> occupancy,
                                   const Bitboard checkedMask, const Bitboard pinned) const {
    const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];

    Bitboard sbb = bitboards_[our][piece];
    if constexpr (piece == kKnight) {
      sbb &= ~pinned; // Pinned knight can never move.
    }

    for (; sbb; sbb = popPiece(sbb)) {
      const Square srce = peekPiece(sbb);

      // Don't attack our pieces, and must block or capture checker if there's any.
      Bitboard dbb = ~occupancy[our] & checkedMask;

      // Restrict the piece movement in the pinned direction.
      if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
        if (isSquareSet(pinned, srce)) {
          dbb &= kLineOfSightMasks[kingSq][srce];
        }
      }

      // Generate actual moves.
      if constexpr (piece == kKnight) {
        dbb &= getAttack<kKnight>(srce);
      } else {
        dbb &= getAttack<piece>(srce, bothOccupancy);
      }

      for (; dbb; dbb = popPiece(dbb)) {
        Square dest = peekPiece(dbb);
        Receiver::acceptMove(*this, Move(srce, dest, piece));
      }
    }
  }

public:
  constexpr Color getColor() const {
    return color_;
  }

  template <Color our, typename Receiver>
  constexpr void enumerateMoves() const {
    constexpr Color their = getOtherColor(our);
    const Square kingSq = peekPiece(bitboards_[our][kKing]);
    const std::array<Bitboard, kColorSize> occupancy = {
      bitboards_[kWhite][kPawn] | bitboards_[kWhite][kKnight] | bitboards_[kWhite][kBishop] | bitboards_[kWhite][kRook] | bitboards_[kWhite][kQueen] | bitboards_[kWhite][kKing],
      bitboards_[kBlack][kPawn] | bitboards_[kBlack][kKnight] | bitboards_[kBlack][kBishop] | bitboards_[kBlack][kRook] | bitboards_[kBlack][kQueen] | bitboards_[kBlack][kKing],
    };
    const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];
    const Bitboard checkedMask = getCheckedMask<our>(kingSq, bothOccupancy);
    const Bitboard pinnedMask = getPinnedMask<our>(kingSq, occupancy);

    // Knight, Bishop, Rook, Queen Moves
    getPieceMove<our, kKnight, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
    getPieceMove<our, kBishop, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
    getPieceMove<our, kRook, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
    getPieceMove<our, kQueen, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
    
    // Pawn Moves
    {
      // Left Attack
      for (Bitboard dbb = (our == kWhite ? shiftUpLeft(bitboards_[our][kPawn]) : shiftDownLeft(bitboards_[our][kPawn])) & occupancy[their] & checkedMask;
           dbb;
           dbb = popPiece(dbb)) {

        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDownRight(dest) : squareUpRight(dest));
        if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kKnight));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kBishop));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kRook));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kQueen));
          } else {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn));
          }
        }
      }

      // Right Attack
      for (Bitboard dbb = (our == kWhite ? shiftUpRight(bitboards_[our][kPawn]) : shiftDownRight(bitboards_[our][kPawn])) & occupancy[their] & checkedMask;
           dbb;
           dbb = popPiece(dbb)) {

        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDownLeft(dest) : squareUpLeft(dest));
        if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kKnight));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kBishop));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kRook));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kQueen));
          } else {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn));
          }
        }
      }

      // Push Forward
      const Bitboard singlePushBB = (our == kWhite ? shiftUp(bitboards_[our][kPawn]) : shiftDown(bitboards_[our][kPawn])) & ~bothOccupancy;
      for (Bitboard dbb = singlePushBB & checkedMask;
           dbb;
           dbb = popPiece(dbb)) {
        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDown(dest) : squareUp(dest));
        if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kKnight));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kBishop));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kRook));
            Receiver::acceptMove(*this, Move(srce, dest, kPawn, kQueen));
          } else {
            Receiver::acceptMove(*this, Move(srce, dest, kPawn));
          }
        }
      }

      // Push Twice
      const Bitboard doublePushBB = (our == kWhite ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
      for (Bitboard dbb = doublePushBB & checkedMask;
           dbb;
           dbb = popPiece(dbb)) {
        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
        if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          Receiver::acceptMove(*this, Move(srce, dest, kPawn, 0, Move::kDoublePushFlag));
        }
      }

      // Enpassant
      if (enpassant_ != NO_SQUARE) {

        // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
        Square capturedSq = (their == kWhite ? squareUp(enpassant_) : squareDown(enpassant_));
        if (isSquareSet(checkedMask, enpassant_) || isSquareSet(checkedMask, capturedSq)) {
          for (Bitboard sbb = getAttack<kPawn, their>(enpassant_) & bitboards_[our][kPawn];
               sbb;
               sbb = popPiece(sbb)) {
            Square srce = peekPiece(sbb);
            Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
            Bitboard discoverAttack = getAttack<kBishop>(kingSq, pseudoOccupancy) & (bitboards_[their][kBishop] | bitboards_[their][kQueen]) |
              getAttack<kRook>(kingSq, pseudoOccupancy) & (bitboards_[their][kRook] | bitboards_[their][kQueen]);
            if (!discoverAttack) {
              Receiver::acceptMove(*this, Move(srce, enpassant_, kPawn, 0, Move::kEnpassantFlag));
            }
          }
        }
      }
    }

    // King Moves
    const Bitboard attackedMask = getAttackedMask<our>(bothOccupancy);

    // King Walk
    for (Bitboard bb = getAttack<kKing>(kingSq) & ~occupancy[our] & ~attackedMask;
          bb;
          bb = popPiece(bb)) {
      Square dest = peekPiece(bb);
      Receiver::acceptMove(*this, Move(kingSq, dest, kKing));
    }

    // King Castling
    if ((castlePermission_ & kKingCastlePermission[our]) == kKingCastlePermission[our] &&  // Check castle permission
        (bothOccupancy & kKingCastleOccupancy[our]) == 0 &&                                // Check castle blocker
        (attackedMask & kKingCastleSafety[our]) == 0) {                                        // Check castle attacked squares
      if constexpr (our == kWhite) {
        Receiver::acceptMove(*this, Move(E1, G1, kKing, 0, Move::kKingSideCastleFlag));
      } else {
        Receiver::acceptMove(*this, Move(E8, G8, kKing, 0, Move::kKingSideCastleFlag));
      }
    }

    // Queen Castling
    if ((castlePermission_ & kQueenCastlePermission[our]) == kQueenCastlePermission[our] && // Check castle permission
        (bothOccupancy & kQueenCastleOccupancy[our]) == 0 &&                                // Check castle blocker
        (attackedMask & kQueenCastleSafety[our]) == 0) {                                        // Check castle attacked squares
      if constexpr (our == kWhite) {
        Receiver::acceptMove(*this, Move(E1, C1, kKing, 0, Move::kQueenSideCastleFlag));
      } else {
        Receiver::acceptMove(*this, Move(E8, C8, kKing, 0, Move::kQueenSideCastleFlag));
      }
    }
  }

  static BoardState fromFEN(const std::string& fen);
  friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
};
static_assert(std::is_trivial_v<BoardState>, "BoardState is not POD type, may affect performance");
