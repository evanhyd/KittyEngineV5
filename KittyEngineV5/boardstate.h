#pragma once
#include "bitboard.h"
#include "move.h"
#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace bb {
  // BOARD STATE NODE TYPE //
  struct NodeType {
    Color color;
  };

  // BOARD STATE //
  class BoardState {
  private:
    std::array<std::array<Bitboard, kPieceSize>, kColorSize> bitboards_;
    Bitboard castlePermission_;
    Color color_;
    Square enpassant_;
    int32_t halfmove_;
    int32_t fullmove_;

    template <Color ally>
    constexpr void addPawnMove(MoveList& moves, Square srce, Square dest, uint32_t flags = 0) const {
      if (getSquareRank(dest) == kPromotionRank[ally]) {
        moves.push(Move(srce, dest, kPawn, kKnight, flags));
        moves.push(Move(srce, dest, kPawn, kBishop, flags));
        moves.push(Move(srce, dest, kPawn, kRook, flags));
        moves.push(Move(srce, dest, kPawn, kQueen, flags));
      } else {
        moves.push(Move(srce, dest, kPawn, kNoPiece, flags));
      }
    }

    template <Color ally, bool kingSide>
    constexpr void addCastlingMove(MoveList& moves, Bitboard bothOccupancy, Bitboard attackedMask) const {
      constexpr Bitboard permission = kingSide ? kKingCastlePermission[ally] : kQueenCastlePermission[ally];
      constexpr Bitboard blockers = kingSide ? kKingCastleOccupancy[ally] : kQueenCastleOccupancy[ally];
      constexpr Bitboard safety = kingSide ? kKingCastleSafety[ally] : kQueenCastleSafety[ally];
      if ((castlePermission_ & permission) == permission &&
          (bothOccupancy & blockers) == 0 &&
          (attackedMask & safety) == 0) {
        constexpr Square srce = ally == kWhite ? E1 : E8;
        constexpr Square dest = ally == kWhite ? (kingSide ? G1 : C1) : (kingSide ? G8 : C8);
        moves.push(Move(srce, dest, kKing, kNoPiece, Move::kCastlingFlag));
      }
    }

  public:
    explicit BoardState() noexcept;
    explicit BoardState(std::string_view fen);
    void setPosition(std::string_view fen);

    constexpr Bitboard getPieces(Color color, Piece piece) const noexcept { return bitboards_[color][piece]; }
    constexpr Color getColorToMove() const noexcept { return color_; }
    constexpr Bitboard getCastlingRights() const noexcept { return castlePermission_; }
    constexpr Square getEnpassantSquare() const noexcept { return enpassant_; }
    constexpr int32_t getHalfmoveClock() const noexcept { return halfmove_; }
    constexpr int32_t getFullmoveNumber() const noexcept { return fullmove_; }

    constexpr Bitboard getOccupancy(Color color) const noexcept {
      return bitboards_[color][kPawn] | bitboards_[color][kKnight] |
        bitboards_[color][kBishop] | bitboards_[color][kRook] |
        bitboards_[color][kQueen] | bitboards_[color][kKing];
    }

    constexpr std::optional<std::tuple<Color, Piece>> getPieceAt(Square square) const {
      for (Color color = kWhite; color < kColorSize; ++color) {
        for (Piece piece = kPawn; piece < kPieceSize; ++piece) {
          if (isSquareSet(bitboards_[color][piece], square)) {
            return std::tuple<Color, Piece>{color, piece};
          }
        }
      }
      return std::nullopt;
    }

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

    // Filter for non-king move destinations: all squares when there is no check,
    // blocking or capturing squares for one checker, and no squares for double check.
    template <Color ally>
    constexpr Bitboard getCheckEvasionMask(Square kingSq, Bitboard bothOccupancy) const {
      constexpr Color enemy = getOtherColor(ally);

      Bitboard evasionMask = ~Bitboard{};
      Bitboard checkers = (getAttack<kPawn, ally>(kingSq) & bitboards_[enemy][kPawn]) |
        (getAttack<kKnight>(kingSq) & bitboards_[enemy][kKnight]) |
        (getAttack<kBishop>(kingSq, bothOccupancy) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen])) |
        (getAttack<kRook>(kingSq, bothOccupancy) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]));

      for (; checkers; checkers = popPiece(checkers)) {
        const Square sq = peekPiece(checkers);
        evasionMask &= setSquare(kSquareBetweenMasks[kingSq][sq], sq);
      }
      return evasionMask;
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
      MoveList& moves,
      const Square kingSq,
      const std::array<Bitboard, kColorSize> occupancy,
      const Bitboard evasionMask, 
      const Bitboard pinnedMask) const {

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
        Bitboard dbb = ~occupancy[ally] & evasionMask;
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
          moves.push(Move(srce, dest, piece, kNoPiece, Move::kCaptureFlag));
        }

        // Generate non-capture moves.
        for (Bitboard nonCaptureDbb = dbb & ~occupancy[enemy]; nonCaptureDbb; nonCaptureDbb = popPiece(nonCaptureDbb)) {
          Square dest = peekPiece(nonCaptureDbb);
          moves.push(Move(srce, dest, piece));
        }
      }
    }

    template <Color ally>
    constexpr bool isInCheck() const {
      const Square kingSq = peekPiece(bitboards_[ally][kKing]);
      const Bitboard bothOccupancy = getOccupancy(kWhite) | getOccupancy(kBlack);
      return isSquareSet(getAttackedMask<ally>(bothOccupancy), kingSq);
    }

    template <Color ally>
    constexpr void generateMoves(MoveList& moves) const {
      assert(color_ == ally);
      constexpr Color enemy = getOtherColor(ally);
      const Square kingSq = peekPiece(bitboards_[ally][kKing]);
      const std::array<Bitboard, kColorSize> occupancy = {getOccupancy(kWhite), getOccupancy(kBlack)};
      const Bitboard bothOccupancy = occupancy[kWhite] | occupancy[kBlack];
      const Bitboard evasionMask = getCheckEvasionMask<ally>(kingSq, bothOccupancy);
      const Bitboard pinnedMask = getPinnedMask<ally>(kingSq, occupancy);

      // Knight, Bishop, Rook, Queen Moves
      getPieceMove<ally, kKnight>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, kBishop>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, kRook>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, kQueen>(moves, kingSq, occupancy, evasionMask, pinnedMask);

      // Pawn Moves
      {
        // Left capture
        for (Bitboard dbb = (ally == kWhite ? shiftUpLeft(bitboards_[ally][kPawn]) : shiftDownLeft(bitboards_[ally][kPawn])) & occupancy[enemy] & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDownRight(dest) : squareUpRight(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally>(moves, srce, dest, Move::kCaptureFlag);
          }
        }

        // Right capture
        for (Bitboard dbb = (ally == kWhite ? shiftUpRight(bitboards_[ally][kPawn]) : shiftDownRight(bitboards_[ally][kPawn])) & occupancy[enemy] & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDownLeft(dest) : squareUpLeft(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally>(moves, srce, dest, Move::kCaptureFlag);
          }
        }

        // Push Forward
        const Bitboard singlePushBB = (ally == kWhite ? shiftUp(bitboards_[ally][kPawn]) : shiftDown(bitboards_[ally][kPawn])) & ~bothOccupancy;
        for (Bitboard dbb = singlePushBB & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDown(dest) : squareUp(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally>(moves, srce, dest);
          }
        }

        // Push Twice
        const Bitboard doublePushBB = (ally == kWhite ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
        for (Bitboard dbb = doublePushBB & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == kWhite ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            moves.push(Move(srce, dest, kPawn, kNoPiece, Move::kDoublePushFlag));
          }
        }

        // Enpassant
        if (enpassant_ != kNoSquare) {

          // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
          Square capturedSq = (enemy == kWhite ? squareUp(enpassant_) : squareDown(enpassant_));
          if (isSquareSet(evasionMask, enpassant_) || isSquareSet(evasionMask, capturedSq)) {
            for (Bitboard sbb = getAttack<kPawn, enemy>(enpassant_) & bitboards_[ally][kPawn];
                 sbb;
                 sbb = popPiece(sbb)) {
              Square srce = peekPiece(sbb);
              Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
              Bitboard discoverAttack = getAttack<kBishop>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kBishop] | bitboards_[enemy][kQueen]) |
                getAttack<kRook>(kingSq, pseudoOccupancy) & (bitboards_[enemy][kRook] | bitboards_[enemy][kQueen]);
              if (!discoverAttack) {
                moves.push(Move(srce, enpassant_, kPawn, kNoPiece, Move::kEnpassantFlag));
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
          moves.push(Move(kingSq, dest, kKing, kNoPiece, Move::kCaptureFlag));
        } else {
          moves.push(Move(kingSq, dest, kKing));
        }
      }

      addCastlingMove<ally, true>(moves, bothOccupancy, attackedMask);
      addCastlingMove<ally, false>(moves, bothOccupancy, attackedMask);
    }


    // Save only the irreversible state; piece moves can be reversed from Move.
    template <Color ally>
    constexpr MoveUndo makeMove(Move move) noexcept {
      assert(color_ == ally);
      constexpr Color enemy = getOtherColor(ally);
      const Square srce = move.getSource();
      const Square dest = move.getDest();
      const Piece movedPiece = move.getMovedPiece();
      const Piece promotion = move.getPromotedPieceType();
      MoveUndo undo{castlePermission_, enpassant_, halfmove_, fullmove_, kNoPiece};

      if (move.isEnpassant()) {
        const Square capturedSq = (ally == kWhite ? squareDown(dest) : squareUp(dest));
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
      enpassant_ = move.isDoublePush() ? (ally == kWhite ? squareUp(srce) : squareDown(srce)) : kNoSquare;
      halfmove_ = (movedPiece == kPawn || undo.capturedPiece != kNoPiece) ? 0 : halfmove_ + 1;
      if constexpr (ally == kBlack) {
        ++fullmove_;
      }
      color_ = enemy;
      return undo;
    }

    template <Color ally>
    constexpr void unmakeMove(Move move, const MoveUndo& undo) noexcept {
      assert(color_ == getOtherColor(ally));
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
      color_ = ally;
    }
  };
  static_assert(std::is_trivially_copyable_v<BoardState>, "BoardState must remain cheap to copy");
}
