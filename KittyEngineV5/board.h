#pragma once
#include "move.h"
#include <array>
#include <set>

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
  std::array<std::array<Bitboard, kPieceSize>, kColorSize> bitboards_;
  Bitboard castlePermission_;
  Square enpassant_;
  uint32_t halfmove_;
  uint32_t fullmove_;
  Color color_;

  // Return a bitboard containing squares attacked by their pieces.
  template <Color our>
  constexpr Bitboard getAttacked(Bitboard bothOccupancy) const {
    constexpr Color their = getOtherColor(our);

    // If king blocks the attack ray, then it may incorrectly move backward illegally.
    // Consider the move: r...K... -> r....K..
    const Bitboard occupancy = bothOccupancy & ~bitboards_[our][kKing];

    // Calculate the attack masks.
    Bitboard attacked = (their == kWhite ?
                         shiftUpLeft(bitboards_[their][kPawn]) | shiftUpRight(bitboards_[their][kPawn]) :
                         shiftDownLeft(bitboards_[their][kPawn]) | shiftDownRight(bitboards_[their][kPawn]));

    attacked |= getAttack<kKing>(peekPiece(bitboards_[their][kKing]));

    for (Bitboard bb = bitboards_[their][kKnight]; bb; bb = popPiece(bb)) {
      attacked |= getAttack<kKnight>(peekPiece(bb));
    }
    for (Bitboard bb = bitboards_[their][kBishop]; bb; bb = popPiece(bb)) {
      attacked |= getAttack<kBishop>(peekPiece(bb), occupancy);
    }
    for (Bitboard bb = bitboards_[their][kRook]; bb; bb = popPiece(bb)) {
      attacked |= getAttack<kRook>(peekPiece(bb), occupancy);
    }
    for (Bitboard bb = bitboards_[their][kQueen]; bb; bb = popPiece(bb)) {
      attacked |= getAttack<kQueen>(peekPiece(bb), occupancy);
    }
    return attacked;
  }

  // Return a bitboard containing the intersection of all attacks.
  // Must block the attack or capture the attackers.
  template <Color our>
  constexpr Bitboard getCheckMask(Square kingSq, Bitboard bothOccupancy) const {
    constexpr Color their = getOtherColor(our);

    Bitboard checkmask = ~Bitboard{};
    Bitboard checkers = (getAttack<kPawn, our>(kingSq) & bitboards_[their][kPawn]) |
                        (getAttack<kKnight>(kingSq) & bitboards_[their][kKnight]) |
                        (getAttack<kBishop>(kingSq, bothOccupancy) & (bitboards_[their][kBishop] | bitboards_[their][kQueen])) |
                        (getAttack<kRook>(kingSq, bothOccupancy) & (bitboards_[their][kRook] | bitboards_[their][kQueen]));

    for (;checkers; checkers = popPiece(checkers)) {
      const Square sq = peekPiece(checkers);
      checkmask &= setSquare(kSquareBetweenMasks[kingSq][sq], sq);
    }
    return checkmask;
  }

  // Return a bitboard containing our pieces that are pinned.
  template <Color our>
  constexpr Bitboard getPinned(Square kingSq , const std::array<Bitboard, kColorSize> occupancy) const {
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

  template <Color our, Piece piece>
  constexpr void getLegalPieceMove(MoveList& moveList, const Square kingSq, const std::array<Bitboard, kColorSize> occupancy,
                                   const Bitboard checkmask, const Bitboard pinned) const {
    const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];

    Bitboard sbb = bitboards_[our][piece];
    if constexpr (piece == kKnight) {
      sbb &= ~pinned; // Pinned knight can never move.
    }

    for (; sbb; sbb = popPiece(sbb)) {
      const Square srce = peekPiece(sbb);

      // Don't attack our pieces, and must block or capture checker if there's any.
      Bitboard dbb = ~occupancy[our] & checkmask;

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
        moveList.push(Move(srce, dest, piece));
      }
    }
  }

public:
  constexpr Color getColor() const {
    return color_;
  }

  template <Color our>
  constexpr void getLegalMove(MoveList& moveList) const {
    constexpr Color their = getOtherColor(our);
    const Square kingSq = peekPiece(bitboards_[our][kKing]);
    const std::array<Bitboard, kColorSize> occupancy = {
      bitboards_[kWhite][kPawn] | bitboards_[kWhite][kKnight] | bitboards_[kWhite][kBishop] | bitboards_[kWhite][kRook] | bitboards_[kWhite][kQueen] | bitboards_[kWhite][kKing],
      bitboards_[kBlack][kPawn] | bitboards_[kBlack][kKnight] | bitboards_[kBlack][kBishop] | bitboards_[kBlack][kRook] | bitboards_[kBlack][kQueen] | bitboards_[kBlack][kKing],
    };
    const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];
    const Bitboard checkmask = getCheckMask<our>(kingSq, bothOccupancy);
    const Bitboard pinned = getPinned<our>(kingSq, occupancy);

    // Knight, Bishop, Rook, Queen Moves
    getLegalPieceMove<our, kKnight>(moveList, kingSq, occupancy, checkmask, pinned);
    getLegalPieceMove<our, kBishop>(moveList, kingSq, occupancy, checkmask, pinned);
    getLegalPieceMove<our, kRook>(moveList, kingSq, occupancy, checkmask, pinned);
    getLegalPieceMove<our, kQueen>(moveList, kingSq, occupancy, checkmask, pinned);
    
    // Pawn Moves
    {
      // Left Attack
      for (Bitboard dbb = (our == kWhite ? shiftUpLeft(bitboards_[our][kPawn]) : shiftDownLeft(bitboards_[our][kPawn])) & occupancy[their] & checkmask;
           dbb;
           dbb = popPiece(dbb)) {

        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDownRight(dest) : squareUpRight(dest));
        if (!isSquareSet(pinned, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            moveList.push(Move(srce, dest, kPawn, kKnight));
            moveList.push(Move(srce, dest, kPawn, kBishop));
            moveList.push(Move(srce, dest, kPawn, kRook));
            moveList.push(Move(srce, dest, kPawn, kQueen));
          } else {
            moveList.push(Move(srce, dest, kPawn));
          }
        }
      }

      // Right Attack
      for (Bitboard dbb = (our == kWhite ? shiftUpRight(bitboards_[our][kPawn]) : shiftDownRight(bitboards_[our][kPawn])) & occupancy[their] & checkmask;
           dbb;
           dbb = popPiece(dbb)) {

        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDownLeft(dest) : squareUpLeft(dest));
        if (!isSquareSet(pinned, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            moveList.push(Move(srce, dest, kPawn, kKnight));
            moveList.push(Move(srce, dest, kPawn, kBishop));
            moveList.push(Move(srce, dest, kPawn, kRook));
            moveList.push(Move(srce, dest, kPawn, kQueen));
          } else {
            moveList.push(Move(srce, dest, kPawn));
          }
        }
      }

      // Push Forward
      const Bitboard singlePushBB = (our == kWhite ? shiftUp(bitboards_[our][kPawn]) : shiftDown(bitboards_[our][kPawn])) & ~bothOccupancy;
      for (Bitboard dbb = singlePushBB & checkmask;
           dbb;
           dbb = popPiece(dbb)) {
        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDown(dest) : squareUp(dest));
        if (!isSquareSet(pinned, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
          if (getSquareRank(dest) == kPromotionRank[our]) {
            moveList.push(Move(srce, dest, kPawn, kKnight));
            moveList.push(Move(srce, dest, kPawn, kBishop));
            moveList.push(Move(srce, dest, kPawn, kRook));
            moveList.push(Move(srce, dest, kPawn, kQueen));
          } else {
            moveList.push(Move(srce, dest, kPawn));
          }
        }
      }

      // Push Twice
      const Bitboard doublePushBB = (our == kWhite ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
      for (Bitboard dbb = doublePushBB & checkmask;
           dbb;
           dbb = popPiece(dbb)) {
        const Square dest = peekPiece(dbb);
        const Square srce = (our == kWhite ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
        if (!isSquareSet(pinned, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            moveList.push(Move(srce, dest, kPawn, 0, Move::kDoublePushFlag));
        }
      }

      // Enpassant
      if (enpassant_ != NO_SQUARE) {

        // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
        Square capturedSq = (their == kWhite ? squareUp(enpassant_) : squareDown(enpassant_));
        if (isSquareSet(checkmask, enpassant_) || isSquareSet(checkmask, capturedSq)) {
          for (Bitboard sbb = getAttack<kPawn, their>(enpassant_) & bitboards_[our][kPawn];
               sbb;
               sbb = popPiece(sbb)) {
            Square srce = peekPiece(sbb);
            Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
            Bitboard discoverAttack = getAttack<kBishop>(kingSq, pseudoOccupancy) & (bitboards_[their][kBishop] | bitboards_[their][kQueen]) |
              getAttack<kRook>(kingSq, pseudoOccupancy) & (bitboards_[their][kRook] | bitboards_[their][kQueen]);
            if (!discoverAttack) {
              moveList.push(Move(srce, enpassant_, kPawn, 0, Move::kEnpassantFlag));
            }
          }
        }
      }
    }


    // King Moves
    const Bitboard attacked = getAttacked<our>(bothOccupancy);

    // King Walk
    for (Bitboard bb = getAttack<kKing>(kingSq) & ~occupancy[our] & ~attacked;
          bb;
          bb = popPiece(bb)) {
      Square dest = peekPiece(bb);
      moveList.push(Move(kingSq, dest, kKing));
    }

    // King Castling
    if ((castlePermission_ & kKingCastlePermission[our]) == kKingCastlePermission[our] &&  // Check castle permission
        (bothOccupancy & kKingCastleOccupancy[our]) == 0 &&                                // Check castle blocker
        (attacked & kKingCastleSafety[our]) == 0) {                                        // Check castle attacked squares
      if constexpr (our == kWhite) {
        moveList.push(Move(E1, G1, kKing, 0, Move::kKingSideCastleFlag));
      } else {
        moveList.push(Move(E8, G8, kKing, 0, Move::kKingSideCastleFlag));
      }
    }

    // Queen Castling
    if ((castlePermission_ & kQueenCastlePermission[our]) == kQueenCastlePermission[our] && // Check castle permission
        (bothOccupancy & kQueenCastleOccupancy[our]) == 0 &&                                // Check castle blocker
        (attacked & kQueenCastleSafety[our]) == 0) {                                        // Check castle attacked squares
      if constexpr (our == kWhite) {
        moveList.push(Move(E1, C1, kKing, 0, Move::kQueenSideCastleFlag));
      } else {
        moveList.push(Move(E8, C8, kKing, 0, Move::kQueenSideCastleFlag));
      }
    }
  }

  static BoardState fromFEN(const std::string& fen);
  friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
  friend class Board;
  friend class PerftDriver;
};
static_assert(std::is_trivial_v<BoardState>, "BoardState is not POD type, may affect performance");


///////////////////////////////////////////////////////
//                 CHESS BOARD
///////////////////////////////////////////////////////
class Board {
public:
  template <Color our>
  static constexpr BoardState makeMove(BoardState state, Move move) {
    constexpr Color their = getOtherColor(our);
    const Square srce = move.getSourceSquare();
    const Square dest = move.getDestinationSquare();
    const Piece movedPiece = move.getMovedPiece();

    // Move the square.
    state.bitboards_[our][movedPiece] = moveSquare(state.bitboards_[our][movedPiece], srce, dest);
    state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], dest);
    state.bitboards_[their][kKnight] = unsetSquare(state.bitboards_[their][kKnight], dest);
    state.bitboards_[their][kBishop] = unsetSquare(state.bitboards_[their][kBishop], dest);
    state.bitboards_[their][kRook] = unsetSquare(state.bitboards_[their][kRook], dest);
    state.bitboards_[their][kQueen] = unsetSquare(state.bitboards_[their][kQueen], dest);

    // Reset enpassant square
    const Square enpassantSq = state.enpassant_;
    state.enpassant_ = NO_SQUARE;

    // Update castle occupancy.
    state.castlePermission_ = unsetSquare(unsetSquare(state.castlePermission_, srce), dest);

    // Update half move and full move.
    ++state.halfmove_;
    /*if (move.isCaptured()) {
      state.halfmove_ = 0;
    }*/

    if constexpr (our == kBlack) {
      ++state.fullmove_;
    }

    if (movedPiece == kPawn) {
      state.halfmove_ = 0;

      if (move.isEnpassant()) {
        if constexpr (their == kWhite) {
          state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], squareUp(enpassantSq));
        } else {
          state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], squareDown(enpassantSq));
        }
      } else if (move.isDoublePush()) {
        if constexpr (our == kWhite) {
          state.enpassant_ = squareUp(srce);
        } else {
          state.enpassant_ = squareDown(srce);
        }
      } else if (Piece promotedPiece = move.getPromotedPiece(); promotedPiece) {
        state.bitboards_[our][kPawn] = unsetSquare(state.bitboards_[our][kPawn], dest);
        state.bitboards_[our][promotedPiece] = setSquare(state.bitboards_[our][promotedPiece], dest);
      }

    } else if (movedPiece == kKing) {
      if (move.isKingSideCastle()) {
        if constexpr (our == kWhite) {
          state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], H1, F1);
        } else {
          state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], H8, F8);
        }
      } else if (move.isQueenSideCastle()) {
        if constexpr (our == kWhite) {
          state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], A1, D1);
        } else {
          state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], A8, D8);
        }
      }
    }

    state.color_ = getOtherColor(our);
    return state;
  }
};
