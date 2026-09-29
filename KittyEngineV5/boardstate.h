#pragma once
#include "bitboard.h"
#include "move.h"
#include <array>
#include <iosfwd>
#include <string>
#include <type_traits>

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

    struct Undo {
      Bitboard castlePermission;
      Square enpassant;
      uint32_t halfmove;
      uint32_t fullmove;
      Piece capturedPiece;
    };


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

    template <Color ally, Piece piece>
    constexpr void getPieceMove(
      MoveList& moveList,
      const Square kingSq,
      const std::array<Bitboard, kColorSize> occupancy,
      const Bitboard checkedMask, const Bitboard pinnedMask) const {

      const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];
      constexpr Color enemy = getOtherColor(ally);
      Bitboard sbb = bitboards_[ally][piece];
      if constexpr (piece == kKnight) {
        sbb &= ~pinnedMask; // Pinned knight can never move.
      }

      for (; sbb; sbb = popPiece(sbb)) {
        const Square srce = peekPiece(sbb);

        // Create filter mask.
        // Don't attack ally pieces.
        // Must block or capture checker if there's any.
        // Restrict the piece movement in the pinned direction.
        Bitboard dbb = ~occupancy[ally] & checkedMask;
        if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
          if (isSquareSet(pinnedMask, srce)) {
            dbb &= kLineOfSightMasks[kingSq][srce];
          }
        }

        // Combine the filter mask with the piece attack mask.
        if constexpr (piece == kKnight) {
          dbb &= getAttack<kKnight>(srce);
        } else {
          dbb &= getAttack<piece>(srce, bothOccupancy);
        }

        // Generate capture moves.
        for (Bitboard captureDbb = dbb & occupancy[enemy]; captureDbb; captureDbb = popPiece(captureDbb)) {
          Square dest = peekPiece(captureDbb);
          moveList.push(Move(srce, dest, piece, kNoPiece, Move::kCaptureFlag));
        }

        // Generate non-capture moves.
        for (Bitboard nonCaptureDbb = dbb & ~occupancy[enemy]; nonCaptureDbb; nonCaptureDbb = popPiece(nonCaptureDbb)) {
          Square dest = peekPiece(nonCaptureDbb);
          moveList.push(Move(srce, dest, piece));
        }
      }
    }

  public:

    template <Color ally>
    constexpr void generateMoves(MoveList& moveList) const {
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
      getPieceMove<ally, kKnight>(moveList, kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kBishop>(moveList, kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kRook>(moveList, kingSq, occupancy, checkedMask, pinnedMask);
      getPieceMove<ally, kQueen>(moveList, kingSq, occupancy, checkedMask, pinnedMask);

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
              moveList.push(Move(srce, dest, kPawn, kKnight, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kBishop, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kRook, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kQueen, Move::kCaptureFlag));
            } else {
              moveList.push(Move(srce, dest, kPawn, kNoPiece, Move::kCaptureFlag));
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
              moveList.push(Move(srce, dest, kPawn, kKnight, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kBishop, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kRook, Move::kCaptureFlag));
              moveList.push(Move(srce, dest, kPawn, kQueen, Move::kCaptureFlag));
            } else {
              moveList.push(Move(srce, dest, kPawn, kNoPiece, Move::kCaptureFlag));
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
        const Bitboard doublePushBB = (ally == kWhite ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
        for (Bitboard dbb = doublePushBB & checkedMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            moveList.push(Move(srce, dest, kPawn, kNoPiece, Move::kDoublePushFlag));
          }
        }

        // Enpassant
        if (enpassant_ != NO_SQUARE) {

          // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
          Square capturedSq = (enemy == kWhite ? squareUp(enpassant_) : squareDown(enpassant_));
          if (isSquareSet(checkedMask, enpassant_) || isSquareSet(checkedMask, capturedSq)) {
            for (Bitboard sbb = getAttack<kPawn, enemy>(enpassant_) & bitboards_[ally][kPawn];
                 sbb;
                 sbb = popPiece(sbb)) {
              Square srce = peekPiece(sbb);
              Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
              Bitboard discoverAttack = getAttack<kBishop>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen]) |
                getAttack<kRook>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]);
              if (!discoverAttack) {
                moveList.push(Move(srce, enpassant_, kPawn, kNoPiece, Move::kEnpassantFlag));
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
        if (isSquareSet(occupancy[enemy], dest)) {
          moveList.push(Move(kingSq, dest, kKing, kNoPiece, Move::kCaptureFlag));
        } else {
          moveList.push(Move(kingSq, dest, kKing));
        }
      }

      // King Castling
      if ((castlePermission_ & kKingCastlePermission[ally]) == kKingCastlePermission[ally] &&  // Check castle permission
          (bothOccupancy & kKingCastleOccupancy[ally]) == 0 &&                                // Check castle blocker
          (attackedMask & kKingCastleSafety[ally]) == 0) {                                        // Check castle attacked squares
        if constexpr (ally == kWhite) {
          moveList.push(Move(E1, G1, kKing, kNoPiece, Move::kCastlingFlag));
        } else {
          moveList.push(Move(E8, G8, kKing, kNoPiece, Move::kCastlingFlag));
        }
      }

      // Queen Castling
      if ((castlePermission_ & kQueenCastlePermission[ally]) == kQueenCastlePermission[ally] && // Check castle permission
          (bothOccupancy & kQueenCastleOccupancy[ally]) == 0 &&                                // Check castle blocker
          (attackedMask & kQueenCastleSafety[ally]) == 0) {                                        // Check castle attacked squares
        if constexpr (ally == kWhite) {
          moveList.push(Move(E1, C1, kKing, kNoPiece, Move::kCastlingFlag));
        } else {
          moveList.push(Move(E8, C8, kKing, kNoPiece, Move::kCastlingFlag));
        }
      }
    }


    // Save only the irreversible state; piece moves can be reversed from Move.
    template <Color ally>
    constexpr Undo makeMove(Move move) noexcept {
      static_assert(ally == kWhite || ally == kBlack);
      constexpr Color enemy = getOtherColor(ally);
      const Square srce = move.getSource();
      const Square dest = move.getDest();
      const Piece movedPiece = move.getMovedPiece();
      const Piece promotion = move.getPromotedPieceType();
      Undo undo{castlePermission_, enpassant_, halfmove_, fullmove_, kNoPiece};

      if (move.isEnpassant()) {
        const Square capturedSq = ally == kWhite ? squareDown(dest) : squareUp(dest);
        bitboards_[enemy][kPawn] = unsetSquare(bitboards_[enemy][kPawn], capturedSq);
        undo.capturedPiece = kPawn;
      } else if (move.isCapture()) {
        for (Piece piece = kPawn; piece <= kQueen; ++piece) {
          if (isSquareSet(bitboards_[enemy][piece], dest)) {
            bitboards_[enemy][piece] = unsetSquare(bitboards_[enemy][piece], dest);
            undo.capturedPiece = piece;
            break;
          }
        }
      }

      bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], srce, dest);
      if (promotion != kNoPiece) {
        bitboards_[ally][kPawn] = unsetSquare(bitboards_[ally][kPawn], dest);
        bitboards_[ally][promotion] = setSquare(bitboards_[ally][promotion], dest);
      }
      if (move.isCastling()) {
        const Square rookFrom = dest > srce ? (ally == kWhite ? H1 : H8) : (ally == kWhite ? A1 : A8);
        const Square rookTo = dest > srce ? dest - 1 : dest + 1;
        bitboards_[ally][kRook] = moveSquare(bitboards_[ally][kRook], rookFrom, rookTo);
      }

      castlePermission_ = unsetSquare(unsetSquare(castlePermission_, srce), dest);
      enpassant_ = move.isDoublePush() ? (ally == kWhite ? squareUp(srce) : squareDown(srce)) : NO_SQUARE;
      halfmove_ = (movedPiece == kPawn || undo.capturedPiece != kNoPiece) ? 0 : halfmove_ + 1;
      if constexpr (ally == kBlack) ++fullmove_;
      return undo;
    }

    template <Color ally>
    constexpr void unmakeMove(Move move, const Undo& undo) noexcept {
      static_assert(ally == kWhite || ally == kBlack);
      constexpr Color enemy = getOtherColor(ally);
      const Square srce = move.getSource();
      const Square dest = move.getDest();
      const Piece promotion = move.getPromotedPieceType();

      if (move.isCastling()) {
        const Square rookFrom = dest > srce ? (ally == kWhite ? H1 : H8) : (ally == kWhite ? A1 : A8);
        const Square rookTo = dest > srce ? dest - 1 : dest + 1;
        bitboards_[ally][kRook] = moveSquare(bitboards_[ally][kRook], rookTo, rookFrom);
      }
      if (promotion != kNoPiece) {
        bitboards_[ally][promotion] = unsetSquare(bitboards_[ally][promotion], dest);
        bitboards_[ally][kPawn] = setSquare(bitboards_[ally][kPawn], srce);
      } else {
        const Piece movedPiece = move.getMovedPiece();
        bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], dest, srce);
      }
      if (undo.capturedPiece != kNoPiece) {
        const Square capturedSq = move.isEnpassant() ? (ally == kWhite ? squareDown(dest) : squareUp(dest)) : dest;
        bitboards_[enemy][undo.capturedPiece] = setSquare(bitboards_[enemy][undo.capturedPiece], capturedSq);
      }
      castlePermission_ = undo.castlePermission;
      enpassant_ = undo.enpassant;
      halfmove_ = undo.halfmove;
      fullmove_ = undo.fullmove;
    }

    static BoardState fromFEN(const std::string& fen, Color& sideToMove);
    static BoardState fromFEN(const std::string& fen) { Color sideToMove; return fromFEN(fen, sideToMove); }
    friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
  };
  static_assert(std::is_trivial_v<BoardState>, "Non-trivial BoardState may affect performance");
}
