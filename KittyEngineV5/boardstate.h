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

    template <Color ally, bool violentOnly = false>
    constexpr void addPawnMove(MoveList& moves, Square srce, Square dest, uint32_t flags = 0) const {
      if (getSquareRank(dest) == kPromotionRank[ally]) {
        moves.push(Move(srce, dest, Pawn, Knight, flags));
        moves.push(Move(srce, dest, Pawn, Bishop, flags));
        moves.push(Move(srce, dest, Pawn, Rook, flags));
        moves.push(Move(srce, dest, Pawn, Queen, flags));
      } else {
        if constexpr (!violentOnly) {
          moves.push(Move(srce, dest, Pawn, NoPiece, flags));
        }
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
        constexpr Square srce = ally == White ? E1 : E8;
        constexpr Square dest = ally == White ? (kingSide ? G1 : C1) : (kingSide ? G8 : C8);
        moves.push(Move(srce, dest, King, NoPiece, Move::kCastlingFlag));
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
      return bitboards_[color][Pawn] | bitboards_[color][Knight] |
        bitboards_[color][Bishop] | bitboards_[color][Rook] |
        bitboards_[color][Queen] | bitboards_[color][King];
    }

    constexpr std::optional<std::tuple<Color, Piece>> getPieceAt(Square square) const {
      for (Color color = White; color < kColorSize; ++color) {
        for (Piece piece = Pawn; piece < kPieceSize; ++piece) {
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
      const Bitboard occupancy = bothOccupancy & ~bitboards_[ally][King];

      // Calculate the attack masks.
      Bitboard attackedMask = (enemy == White ?
                           shiftUpLeft(bitboards_[enemy][Pawn]) | shiftUpRight(bitboards_[enemy][Pawn]) :
                           shiftDownLeft(bitboards_[enemy][Pawn]) | shiftDownRight(bitboards_[enemy][Pawn]));

      attackedMask |= getAttack<King>(peekPiece(bitboards_[enemy][King]));

      for (Bitboard bb = bitboards_[enemy][Knight]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<Knight>(peekPiece(bb));
      }
      for (Bitboard bb = bitboards_[enemy][Bishop]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<Bishop>(peekPiece(bb), occupancy);
      }
      for (Bitboard bb = bitboards_[enemy][Rook]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<Rook>(peekPiece(bb), occupancy);
      }
      for (Bitboard bb = bitboards_[enemy][Queen]; bb; bb = popPiece(bb)) {
        attackedMask |= getAttack<Queen>(peekPiece(bb), occupancy);
      }
      return attackedMask;
    }

    // Filter for non-king move destinations: all squares when there is no check,
    // blocking or capturing squares for one checker, and no squares for double check.
    template <Color ally>
    constexpr Bitboard getCheckEvasionMask(Square kingSq, Bitboard bothOccupancy) const {
      constexpr Color enemy = getOtherColor(ally);

      Bitboard evasionMask = ~Bitboard{};
      Bitboard checkers = (getAttack<Pawn, ally>(kingSq) & bitboards_[enemy][Pawn]) |
        (getAttack<Knight>(kingSq) & bitboards_[enemy][Knight]) |
        (getAttack<Bishop>(kingSq, bothOccupancy) & (bitboards_[enemy][Bishop] | bitboards_[enemy][Queen])) |
        (getAttack<Rook>(kingSq, bothOccupancy) & (bitboards_[enemy][Rook] | bitboards_[enemy][Queen]));

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
      Bitboard sliders = (getAttack<Bishop>(kingSq, occupancy[enemy]) & (bitboards_[enemy][Bishop] | bitboards_[enemy][Queen])) |
        (getAttack<Rook>(kingSq, occupancy[enemy]) & (bitboards_[enemy][Rook] | bitboards_[enemy][Queen]));
      for (; sliders; sliders = popPiece(sliders)) {
        Square sliderSquare = peekPiece(sliders);
        Bitboard blockers = kSquareBetweenMasks[kingSq][sliderSquare] & occupancy[ally];
        if (popPiece(blockers) == 0) {
          pinnedMask |= blockers; // Does NOT handle enpassant edge case.
        }
      }
      return pinnedMask;
    }

    template <Color ally, Piece piece, bool violentOnly = false>
    constexpr void getPieceMove(
      MoveList& moves,
      const Square kingSq,
      const std::array<Bitboard, kColorSize> occupancy,
      const Bitboard evasionMask, 
      const Bitboard pinnedMask) const {

      const Bitboard bothOccupancy = occupancy[White] | occupancy[Black];
      constexpr Color enemy = getOtherColor(ally);
      Bitboard sbb = bitboards_[ally][piece];
      if constexpr (piece == Knight) {
        sbb &= ~pinnedMask; // Pinned knight can never move.
      }

      for (; sbb; sbb = popPiece(sbb)) {
        const Square srce = peekPiece(sbb);

        // Create filter mask.
        // Don't attack ally pieces.
        // Must block or capture checker if there's any.
        // Restrict the piece movement in the pinned direction.
        Bitboard dbb = ~occupancy[ally] & evasionMask;
        if constexpr (piece == Bishop || piece == Rook || piece == Queen) {
          if (isSquareSet(pinnedMask, srce)) {
            dbb &= kLineOfSightMasks[kingSq][srce];
          }
        }

        // Combine the filter mask with the piece attack mask.
        if constexpr (piece == Knight) {
          dbb &= getAttack<Knight>(srce);
        } else {
          dbb &= getAttack<piece>(srce, bothOccupancy);
        }

        // Generate capture moves.
        for (Bitboard captureDbb = dbb & occupancy[enemy]; captureDbb; captureDbb = popPiece(captureDbb)) {
          Square dest = peekPiece(captureDbb);
          moves.push(Move(srce, dest, piece, NoPiece, Move::kCaptureFlag));
        }

        if constexpr (!violentOnly) {
          // Generate non-capture moves.
          for (Bitboard nonCaptureDbb = dbb & ~occupancy[enemy]; nonCaptureDbb; nonCaptureDbb = popPiece(nonCaptureDbb)) {
            Square dest = peekPiece(nonCaptureDbb);
            moves.push(Move(srce, dest, piece));
          }
        }
      }
    }

    template <Color ally>
    constexpr bool isInCheck() const {
      const Square kingSq = peekPiece(bitboards_[ally][King]);
      const Bitboard bothOccupancy = getOccupancy(White) | getOccupancy(Black);
      return isSquareSet(getAttackedMask<ally>(bothOccupancy), kingSq);
    }

    template <Color ally, bool violentOnly = false>
    constexpr void generateMoves(MoveList& moves) const {
      assert(color_ == ally);
      constexpr Color enemy = getOtherColor(ally);
      const Square kingSq = peekPiece(bitboards_[ally][King]);
      const std::array<Bitboard, kColorSize> occupancy = {getOccupancy(White), getOccupancy(Black)};
      const Bitboard bothOccupancy = occupancy[White] | occupancy[Black];
      const Bitboard evasionMask = getCheckEvasionMask<ally>(kingSq, bothOccupancy);
      const Bitboard pinnedMask = getPinnedMask<ally>(kingSq, occupancy);

      // Knight, Bishop, Rook, Queen Moves
      getPieceMove<ally, Knight, violentOnly>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, Bishop, violentOnly>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, Rook, violentOnly>(moves, kingSq, occupancy, evasionMask, pinnedMask);
      getPieceMove<ally, Queen, violentOnly>(moves, kingSq, occupancy, evasionMask, pinnedMask);

      // Pawn Moves
      {
        // Left capture
        for (Bitboard dbb = (ally == White ? shiftUpLeft(bitboards_[ally][Pawn]) : shiftDownLeft(bitboards_[ally][Pawn])) & occupancy[enemy] & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == White ? squareDownRight(dest) : squareUpRight(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally, violentOnly>(moves, srce, dest, Move::kCaptureFlag);
          }
        }

        // Right capture
        for (Bitboard dbb = (ally == White ? shiftUpRight(bitboards_[ally][Pawn]) : shiftDownRight(bitboards_[ally][Pawn])) & occupancy[enemy] & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == White ? squareDownLeft(dest) : squareUpLeft(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally, violentOnly>(moves, srce, dest, Move::kCaptureFlag);
          }
        }

        // Push Forward
        const Bitboard singlePushBB = (ally == White ? shiftUp(bitboards_[ally][Pawn]) : shiftDown(bitboards_[ally][Pawn])) & ~bothOccupancy;
        for (Bitboard dbb = singlePushBB & evasionMask;
             dbb;
             dbb = popPiece(dbb)) {
          const Square dest = peekPiece(dbb);
          const Square srce = (ally == White ? squareDown(dest) : squareUp(dest));
          if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
            addPawnMove<ally, violentOnly>(moves, srce, dest);
          }
        }

        // Push Twice
        if constexpr (!violentOnly) {
          const Bitboard doublePushBB = (ally == White ? (shiftUp(singlePushBB) & kRank4Mask) : (shiftDown(singlePushBB) & kRank5Mask)) & ~bothOccupancy;
          for (Bitboard dbb = doublePushBB & evasionMask;
               dbb;
               dbb = popPiece(dbb)) {
            const Square dest = peekPiece(dbb);
            const Square srce = (ally == White ? squareDown(squareDown(dest)) : squareUp(squareUp(dest)));
            if (!isSquareSet(pinnedMask, srce) || kLineOfSightMasks[kingSq][srce] == kLineOfSightMasks[kingSq][dest]) {
              moves.push(Move(srce, dest, Pawn, NoPiece, Move::kDoublePushFlag));
            }
          }
        }

        // Enpassant
        if (enpassant_ != NoSquare) {

          // Enpassant does 2 things at once. Eliminate the double-pushed pawn checker, and block the enpassant square.
          Square capturedSq = (enemy == White ? squareUp(enpassant_) : squareDown(enpassant_));
          if (isSquareSet(evasionMask, enpassant_) || isSquareSet(evasionMask, capturedSq)) {
            for (Bitboard sbb = getAttack<Pawn, enemy>(enpassant_) & bitboards_[ally][Pawn];
                 sbb;
                 sbb = popPiece(sbb)) {
              Square srce = peekPiece(sbb);
              Bitboard pseudoOccupancy = unsetSquare(moveSquare(bothOccupancy, srce, enpassant_), capturedSq);
              Bitboard discoverAttack = getAttack<Bishop>(kingSq, pseudoOccupancy) & (bitboards_[enemy][Bishop] | bitboards_[enemy][Queen]) |
                getAttack<Rook>(kingSq, pseudoOccupancy) & (bitboards_[enemy][Rook] | bitboards_[enemy][Queen]);
              if (!discoverAttack) {
                moves.push(Move(srce, enpassant_, Pawn, NoPiece, Move::kEnpassantFlag));
              }
            }
          }
        }
      }

      // King Moves
      const Bitboard attackedMask = getAttackedMask<ally>(bothOccupancy);

      // King Walk
      for (Bitboard bb = getAttack<King>(kingSq) & ~occupancy[ally] & ~attackedMask;
            bb;
            bb = popPiece(bb)) {
        Square dest = peekPiece(bb);
        if (isSquareSet(occupancy[enemy], dest)) {
          moves.push(Move(kingSq, dest, King, NoPiece, Move::kCaptureFlag));
        } else {
          if constexpr (!violentOnly) {
            moves.push(Move(kingSq, dest, King));
          }
        }
      }

      if constexpr (!violentOnly) {
        addCastlingMove<ally, true>(moves, bothOccupancy, attackedMask);
        addCastlingMove<ally, false>(moves, bothOccupancy, attackedMask);
      }
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
      MoveUndo undo{castlePermission_, enpassant_, halfmove_, fullmove_, NoPiece};

      if (move.isEnpassant()) {
        const Square capturedSq = (ally == White ? squareDown(dest) : squareUp(dest));
        bitboards_[enemy][Pawn] = unsetSquare(bitboards_[enemy][Pawn], capturedSq);
        undo.capturedPiece = Pawn;
      } else if (move.isCapture()) {
        for (Piece piece = Pawn; piece <= Queen; ++piece) {
          if (isSquareSet(bitboards_[enemy][piece], dest)) {
            bitboards_[enemy][piece] = unsetSquare(bitboards_[enemy][piece], dest);
            undo.capturedPiece = piece;
            break;
          }
        }
      }

      bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], srce, dest);
      if (promotion != NoPiece) {
        bitboards_[ally][Pawn] = unsetSquare(bitboards_[ally][Pawn], dest);
        bitboards_[ally][promotion] = setSquare(bitboards_[ally][promotion], dest);
      }
      if (move.isCastling()) {
        const Square rookFrom = dest > srce ? (ally == White ? H1 : H8) : (ally == White ? A1 : A8);
        const Square rookTo = dest > srce ? dest - 1 : dest + 1;
        bitboards_[ally][Rook] = moveSquare(bitboards_[ally][Rook], rookFrom, rookTo);
      }

      castlePermission_ = unsetSquare(unsetSquare(castlePermission_, srce), dest);
      enpassant_ = move.isDoublePush() ? (ally == White ? squareUp(srce) : squareDown(srce)) : NoSquare;
      halfmove_ = (movedPiece == Pawn || undo.capturedPiece != NoPiece) ? 0 : halfmove_ + 1;
      if constexpr (ally == Black) {
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
        const Square rookFrom = dest > srce ? (ally == White ? H1 : H8) : (ally == White ? A1 : A8);
        const Square rookTo = dest > srce ? dest - 1 : dest + 1;
        bitboards_[ally][Rook] = moveSquare(bitboards_[ally][Rook], rookTo, rookFrom);
      }
      if (promotion != NoPiece) {
        bitboards_[ally][promotion] = unsetSquare(bitboards_[ally][promotion], dest);
        bitboards_[ally][Pawn] = setSquare(bitboards_[ally][Pawn], srce);
      } else {
        const Piece movedPiece = move.getMovedPiece();
        bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], dest, srce);
      }
      if (undo.capturedPiece != NoPiece) {
        const Square capturedSq = move.isEnpassant() ? (ally == White ? squareDown(dest) : squareUp(dest)) : dest;
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
