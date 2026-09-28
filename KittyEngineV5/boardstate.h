#pragma once
#include "bitboard.h"
#include "move.h"
#include <array>

namespace bb {
  // BOARD STATE NODE TYPE //
  struct NodeType {
    Color color;
  };

  // BOARD STATE //
  class BoardState {
  public:
    std::array<std::array<Bitboard, kPieceSize>, kColorSize> bitboards_;
    Bitboard castlePermission_;
    Square enpassant_;
    uint32_t halfmove_;
    uint32_t fullmove_;
    Color color_;

    // Return a bitboard containing squares attacked by enemy pieces.
    template <Color ally>
    constexpr Bitboard getAttackedMask(Bitboard bothOccupancy) const {
      constexpr Color enemy = getOtherColor(ally);

      // If king blocks the attack ray, then it may incorrectly move backward illegally.
      // Consider the move: r...K... -> r....K..
      const Bitboard occupancy = bothOccupancy & ~bitboards_[ally][kKing];

      // Calculate the attack masks.
      Bitboard attackedMask = (enemy == kWhite ?
                           shiftUpLeft(bitboards_[enemy][kPawn]) | shiftUpRight(bitboards_[enemy][kPawn]) :
                           shiftDownLeft(bitboards_[enemy][kPawn]) | shiftDownRight(bitboards_[enemy][kPawn]));

      attackedMask |= getAttack<kKing>(peekPiece(bitboards_[enemy][kKing]));

      for (Bitboard bb = bitboards_[enemy][kKnight]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<kKnight>(peekPiece(bb));
      }
      for (Bitboard bb = bitboards_[enemy][kBishop]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<kBishop>(peekPiece(bb), occupancy);
      }
      for (Bitboard bb = bitboards_[enemy][kRook]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<kRook>(peekPiece(bb), occupancy);
      }
      for (Bitboard bb = bitboards_[enemy][kQueen]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<kQueen>(peekPiece(bb), occupancy);
      }
      return attackedMask;
    }

    // Return a bitboard containing the intersection of all attacks.
    // Must block the attack or capture the attackers.
    template <Color ally>
    constexpr Bitboard getCheckedMask(Square kingSq, Bitboard bothOccupancy) const {
      constexpr Color enemy = getOtherColor(ally);

      Bitboard checkedMask = ~Bitboard{};
      Bitboard checkers = (getAttack<kPawn, ally>(kingSq) & bitboards_[enemy][kPawn]) |
        (getAttack<kKnight>(kingSq) & bitboards_[enemy][kKnight]) |
        (getAttack<kBishop>(kingSq, bothOccupancy) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen])) |
        (getAttack<kRook>(kingSq, bothOccupancy) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]));

      for (; checkers; checkers = popPiece(checkers)) {
        const Square sq = peekPiece(checkers);
        checkedMask &= setSquare(kSquareBetweenMasks[kingSq][sq], sq);
      }
      return checkedMask;
    }

    // Return a bitboard containing ally pieces that are pinned.
    template <Color ally>
    constexpr Bitboard getPinnedMask(Square kingSq, const std::array<Bitboard, kColorSize> occupancy) const {
      constexpr Color enemy = getOtherColor(ally);

      // Get the enemy sliders squares, then check if any ally piece is blocking the attack ray.
      Bitboard pinnedMask{};
      Bitboard sliders = (getAttack<kBishop>(kingSq, occupancy[enemy]) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen])) |
        (getAttack<kRook>(kingSq, occupancy[enemy]) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]));
      for (; sliders; sliders = popPiece(sliders)) {
        Square sliderSquare = peekPiece(sliders);
        Bitboard blockers = kSquareBetweenMasks[kingSq][sliderSquare] & occupancy[ally];
        if (popPiece(blockers) == 0) {
          pinnedMask |= blockers; // Does NOT handle enpassant edge case.
        }
      }
      return pinnedMask;
    }

    template <Color ally, Piece piece, typename Receiver>
    constexpr void getPieceMove(const Square kingSq, const std::array<Bitboard, kColorSize> occupancy,
                                     const Bitboard checkedMask, const Bitboard pinnedMask) const {
      const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];

      Bitboard sbb = bitboards_[ally][piece];
      if constexpr (piece == kKnight) {
        sbb &= ~pinnedMask; // Pinned knight can never move.
      }

      for (; sbb; sbb = popPiece(sbb)) {
        const Square srce = peekPiece(sbb);

        // Don't attack ally pieces, and must block or capture checker if there's any.
        Bitboard dbb = ~occupancy[ally] & checkedMask;

        // Restrict the piece movement in the pinned direction.
        if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
          if (isSquareSet(pinnedMask, srce)) {
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
          Receiver::acceptMove(*this, Move < MoveType{ ally, piece, 0, false, false, false, false } > (srce, dest));
        }
      }
    }

  public:
    constexpr Color getColor() const {
      return color_;
    }

    template <typename NodeType>
    int32_t search(int depth, int32_t alpha, int32_t beta) {

    }

    template <MoveType moveType>
    constexpr BoardState makeMove(Move<moveType> move) const {
      constexpr Color ally = moveType.color;
      constexpr Color enemy = getOtherColor(ally);
      const Square srce = move.srce;
      const Square dest = move.dest;
      BoardState state = *this;
      const Bitboard target = toBitboard(dest);
      const bool isCapture = target & (bitboards_[enemy][kPawn] | bitboards_[enemy][kKnight] |
                                       bitboards_[enemy][kBishop] | bitboards_[enemy][kRook] |
                                       bitboards_[enemy][kQueen]);

      state.bitboards_[ally][moveType.movedPiece] = moveSquare(state.bitboards_[ally][moveType.movedPiece], srce, dest);
      state.bitboards_[enemy][kPawn] = unsetSquare(state.bitboards_[enemy][kPawn], dest);
      state.bitboards_[enemy][kKnight] = unsetSquare(state.bitboards_[enemy][kKnight], dest);
      state.bitboards_[enemy][kBishop] = unsetSquare(state.bitboards_[enemy][kBishop], dest);
      state.bitboards_[enemy][kRook] = unsetSquare(state.bitboards_[enemy][kRook], dest);
      state.bitboards_[enemy][kQueen] = unsetSquare(state.bitboards_[enemy][kQueen], dest);

      const Square enpassantSq = state.enpassant_;
      if constexpr (!moveType.isDoublePush) {
        state.enpassant_ = NO_SQUARE;
      }

      state.castlePermission_ = unsetSquare(unsetSquare(state.castlePermission_, srce), dest);
      state.halfmove_ = (moveType.movedPiece == kPawn || isCapture) ? 0 : halfmove_ + 1;
      if constexpr (ally == kBlack) {
        ++state.fullmove_;
      }

      if constexpr (moveType.movedPiece == kPawn) {
        if constexpr (moveType.isEnpassant) {
          if constexpr (enemy == kWhite) {
            state.bitboards_[enemy][kPawn] = unsetSquare(state.bitboards_[enemy][kPawn], squareUp(enpassantSq));
          } else {
            state.bitboards_[enemy][kPawn] = unsetSquare(state.bitboards_[enemy][kPawn], squareDown(enpassantSq));
          }
        } else if constexpr (moveType.isDoublePush) {
          if constexpr (ally == kWhite) {
            state.enpassant_ = squareUp(srce);
          } else {
            state.enpassant_ = squareDown(srce);
          }
        } else if constexpr (moveType.promotionPiece) {
          state.bitboards_[ally][kPawn] = unsetSquare(state.bitboards_[ally][kPawn], dest);
          state.bitboards_[ally][moveType.promotionPiece] = setSquare(state.bitboards_[ally][moveType.promotionPiece], dest);
        }
      } else if constexpr (moveType.movedPiece == kKing) {
        if constexpr (moveType.isKingSideCastle) {
          if constexpr (ally == kWhite) {
            state.bitboards_[ally][kRook] = moveSquare(state.bitboards_[ally][kRook], H1, F1);
          } else {
            state.bitboards_[ally][kRook] = moveSquare(state.bitboards_[ally][kRook], H8, F8);
          }
        } else if constexpr (moveType.isQueenSideCastle) {
          if constexpr (ally == kWhite) {
            state.bitboards_[ally][kRook] = moveSquare(state.bitboards_[ally][kRook], A1, D1);
          } else {
            state.bitboards_[ally][kRook] = moveSquare(state.bitboards_[ally][kRook], A8, D8);
          }
        }
      }

      state.color_ = enemy;
      return state;
    }

    template <Color ally, typename Receiver>
    constexpr void enumerateMoves() const {
      constexpr Color enemy = getOtherColor(ally);
      const Square kingSq = peekPiece(bitboards_[ally][kKing]);
      const std::array<Bitboard, kColorSize> occupancy = {
        bitboards_[kWhite][kPawn] | bitboards_[kWhite][kKnight] | bitboards_[kWhite][kBishop] | bitboards_[kWhite][kRook] | bitboards_[kWhite][kQueen] | bitboards_[kWhite][kKing],
        bitboards_[kBlack][kPawn] | bitboards_[kBlack][kKnight] | bitboards_[kBlack][kBishop] | bitboards_[kBlack][kRook] | bitboards_[kBlack][kQueen] | bitboards_[kBlack][kKing],
      };
      const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];
      const Bitboard checkedMask = getCheckedMask<ally>(kingSq, bothOccupancy);
      const Bitboard pinnedMask = getPinnedMask<ally>(kingSq, occupancy);

      // Knight, Bishop, Rook, Queen Moves
      getPieceMove<ally, kKnight, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kBishop, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kRook, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kQueen, Receiver>(kingSq, occupancy, checkedMask, pinnedMask);

      // Pawn Moves
      {
        // Left Attack
        for (Bitboard dbb = (ally == kWhite ? shiftUpLeft(bitboards_[ally][kPawn]) : shiftDownLeft(bitboards_[ally][kPawn])) & occupancy[enemy] & checkedMask;
             dbb;
             dbb = popPiece(dbb)) {

          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDownRight(dest) : squareUpRight(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            if (getSquareRank(dest) == kPromotionRank[ally]) {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kKnight, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kBishop, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kRook, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kQueen, false, false, false, false } > (srce, dest));
            } else {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, 0, false, false, false, false } > (srce, dest));
            }
          }
        }

        // Right Attack
        for (Bitboard dbb = (ally == kWhite ? shiftUpRight(bitboards_[ally][kPawn]) : shiftDownRight(bitboards_[ally][kPawn])) & occupancy[enemy] & checkedMask;
             dbb;
             dbb = popPiece(dbb)) {

          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDownLeft(dest) : squareUpLeft(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            if (getSquareRank(dest) == kPromotionRank[ally]) {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kKnight, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kBishop, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kRook, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kQueen, false, false, false, false } > (srce, dest));
            } else {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, 0, false, false, false, false } > (srce, dest));
            }
          }
        }

        // Push Forward
        const Bitboard singlePushBB = (ally == kWhite ? shiftUp(bitboards_[ally][kPawn]) : shiftDown(bitboards_[ally][kPawn])) & ~bothOccupancy;
        for (Bitboard dbb = singlePushBB & checkedMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDown(dest) : squareUp(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            if (getSquareRank(dest) == kPromotionRank[ally]) {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kKnight, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kBishop, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kRook, false, false, false, false } > (srce, dest));
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, kQueen, false, false, false, false } > (srce, dest));
            } else {
              Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, 0, false, false, false, false } > (srce, dest));
            }
          }
        }

        // Push Twice
        const Bitboard doublePushBB = (ally == kWhite ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
        for (Bitboard dbb = doublePushBB & checkedMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, 0, false, true, false, false } > (srce, dest));
          }
        }

        // Enpassant
        if (enpassant_ != NO_SQUARE) {

          // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
          Square capturedSq = (enemy == kWhite ? squareUp(enpassant_) : squareDown(enpassant_));
          if (isSquareSet(checkedMask, enpassant_) || isSquareSet(checkedMask, capturedSq)) {
            for (Bitboard sbb = getAttack<kPawn, enemy>(enpassant_)& bitboards_[ally][kPawn];
                 sbb;
                 sbb = popPiece(sbb)) {
              Square srce = peekPiece(sbb);
              Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
              Bitboard discoverAttack = getAttack<kBishop>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen]) |
                getAttack<kRook>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]);
              if (!discoverAttack) {
                Receiver::acceptMove(*this, Move < MoveType{ ally, kPawn, 0, true, false, false, false } > (srce, enpassant_));
              }
            }
          }
        }
      }

      // King Moves
      const Bitboard attackedMask = getAttackedMask<ally>(bothOccupancy);

      // King Walk
      for (Bitboard bb = getAttack<kKing>(kingSq) & ~occupancy[ally] & ~attackedMask;
            bb;
            bb = popPiece(bb)) {
        Square dest = peekPiece(bb);
        Receiver::acceptMove(*this, Move < MoveType{ ally, kKing, 0, false, false, false, false } > (kingSq, dest));
      }

      // King Castling
      if ((castlePermission_ & kKingCastlePermission[ally]) == kKingCastlePermission[ally] &&  // Check castle permission
          (bothOccupancy & kKingCastleOccupancy[ally]) == 0 &&                                // Check castle blocker
          (attackedMask & kKingCastleSafety[ally]) == 0) {                                        // Check castle attacked squares
        if constexpr (ally == kWhite) {
          Receiver::acceptMove(*this, Move < MoveType{ ally, kKing, 0, false, false, true, false } > (E1, G1));
        } else {
          Receiver::acceptMove(*this, Move < MoveType{ ally, kKing, 0, false, false, true, false } > (E8, G8));
        }
      }

      // Queen Castling
      if ((castlePermission_ & kQueenCastlePermission[ally]) == kQueenCastlePermission[ally] && // Check castle permission
          (bothOccupancy & kQueenCastleOccupancy[ally]) == 0 &&                                // Check castle blocker
          (attackedMask & kQueenCastleSafety[ally]) == 0) {                                        // Check castle attacked squares
        if constexpr (ally == kWhite) {
          Receiver::acceptMove(*this, Move < MoveType{ ally, kKing, 0, false, false, false, true } > (E1, C1));
        } else {
          Receiver::acceptMove(*this, Move < MoveType{ ally, kKing, 0, false, false, false, true } > (E8, C8));
        }
      }
    }

    static BoardState fromFEN(const std::string& fen);
    friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
  };
  static_assert(std::is_trivial_v<BoardState>, "BoardState is not POD type, may affect performance");
}
