#pragma once
#include "bitboard.h"
#include "move.h"
#include "zobrist_hash.h"
#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace bb {
  // BOARD STATE //
  class BoardState {
  private:
    std::array<std::array<Bitboard, kPieceSize>, kSideSize> bitboards_;
    Square enpassant_;
    Side side_;
    CastlePermission castlePermission_;
    int halfmove_;
    int fullmove_;
    ZobristHash zobrist_;

    template <bool isPromotion, bool isCapture = false>
    static constexpr void addPawnMove(MoveList& moves, Square srce, Square dest) noexcept {
      constexpr uint32_t flags = isCapture ? Move::kCaptureFlag : 0;
      if constexpr (isPromotion) {
        moves.push(Move(srce, dest, Pawn, Knight, flags));
        moves.push(Move(srce, dest, Pawn, Bishop, flags));
        moves.push(Move(srce, dest, Pawn, Rook, flags));
        moves.push(Move(srce, dest, Pawn, Queen, flags));
      } else {
        moves.push(Move(srce, dest, Pawn, NoPiece, flags));
      }
    }

    template <bool isPromotion, bool isCapture, typename GetSrce>
    static constexpr void generatePawnMoves(MoveList& moves, Bitboard dbb, Square kingSq, Bitboard pinnedMask, GetSrce&& getSrce) noexcept {
      for (; dbb; dbb = popPiece(dbb)) {
        const Square dest = peekPiece(dbb);
        const Square srce = getSrce(dest);
        if (!isSquareSet(pinnedMask, srce) || isSquareSet(kLineOfSightMasks[kingSq][srce], dest)) {
          addPawnMove<isPromotion, isCapture>(moves, srce, dest);
        }
      }
    }

    template <Side ally, bool kingSide>
    constexpr void addCastlingMove(MoveList& moves, Bitboard bothOccupancy, Bitboard attackedMask) const noexcept {
      constexpr Bitboard permission = []() {
        if constexpr (ally == White) {
          return kingSide ? WhiteKingCastle : WhiteQueenCastle;
        } else {
          return kingSide ? BlackKingCastle : BlackQueenCastle;
        }
      }();
      constexpr Bitboard blockers = kingSide ? kKingCastleOccupancyMasks[ally] : kQueenCastleOccupancyMasks[ally];
      constexpr Bitboard safety = kingSide ? kKingCastleSafetyMasks[ally] : kQueenCastleSafetyMasks[ally];
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

    constexpr Bitboard getPieces(Side side, Piece piece) const noexcept { return bitboards_[side][piece]; }
    constexpr Side getSide() const noexcept { return side_; }
    constexpr CastlePermission getCastlingRights() const noexcept { return castlePermission_; }
    constexpr Square getEnpassantSquare() const noexcept { return enpassant_; }
    constexpr auto getHalfmove() const noexcept { return halfmove_; }
    constexpr auto getFullmove() const noexcept { return fullmove_; }
    constexpr ZobristHash::Hash getHash() const noexcept { return zobrist_.hash(); }

    template <Side side>
    constexpr size_t countMMPiece() const noexcept {
      const Bitboard combined = bitboards_[side][Knight] | bitboards_[side][Bishop] | bitboards_[side][Rook] | bitboards_[side][Queen];
      return countPiece(combined);
    }

    constexpr Bitboard getOccupancy(Side side) const noexcept {
      return bitboards_[side][Pawn] | bitboards_[side][Knight] |
        bitboards_[side][Bishop] | bitboards_[side][Rook] |
        bitboards_[side][Queen] | bitboards_[side][King];
    }

    template <Side side>
    constexpr Piece getPieceAt(Square square) const noexcept {
      for (Piece piece = Pawn; piece < kPieceSize; ++piece) {
        if (isSquareSet(bitboards_[side][piece], square)) {
          return piece;
        }
      }
      return NoPiece;
    }

    constexpr auto getPieceAt(Square square) const noexcept {
      struct SidePiece {
        Side side;
        Piece piece;
      };
      if (const Piece piece = getPieceAt<White>(square); piece != NoPiece) {
        return SidePiece{ White, piece };
      }
      if (const Piece piece = getPieceAt<Black>(square); piece != NoPiece) {
        return SidePiece{ Black, piece };
      }
      return SidePiece{ .piece = NoPiece };
    }

    // Return a bitboard containing squares attacked by enemy pieces.
    template <Side ally>
    constexpr Bitboard getAttackedMask(Bitboard bothOccupancy) const noexcept {
      constexpr Side enemy = getOtherSide(ally);

      // If king blocks the attack ray, then it may incorrectly move backward illegally.
      // Consider the move: r...K... -> r....K..
      const Bitboard occupancy = bothOccupancy & ~bitboards_[ally][King];

      // Calculate the attack masks.
      Bitboard attackedMask;
      if constexpr (enemy == White) {
        attackedMask = shiftUpLeft(bitboards_[enemy][Pawn]) | shiftUpRight(bitboards_[enemy][Pawn]);
      } else {
        attackedMask = shiftDownLeft(bitboards_[enemy][Pawn]) | shiftDownRight(bitboards_[enemy][Pawn]);
      }

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
    template <Side ally>
    constexpr Bitboard getCheckEvasionMask(Square kingSq, Bitboard bothOccupancy) const noexcept {
      constexpr Side enemy = getOtherSide(ally);

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
    template <Side ally>
    constexpr Bitboard getPinnedMask(Square kingSq, const std::array<Bitboard, kSideSize> occupancy) const noexcept {
      constexpr Side enemy = getOtherSide(ally);

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

    template <Side ally, Piece piece, bool violentOnly = false>
    constexpr void getPieceMove(
      MoveList& moves,
      const Square kingSq,
      const std::array<Bitboard, kSideSize> occupancy,
      const Bitboard evasionMask, 
      const Bitboard pinnedMask) const noexcept {

      constexpr Side enemy = getOtherSide(ally);
      const Bitboard bothOccupancy = occupancy[White] | occupancy[Black];
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
        // Combine the filter mask with the piece attack mask.
        if constexpr (piece == Knight) {
          dbb &= getAttack<Knight>(srce);
        } else {
          // Bishop, Rook, Queen.
          if (isSquareSet(pinnedMask, srce)) {
            dbb &= kLineOfSightMasks[kingSq][srce];
          }
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

    template <Side ally>
    constexpr bool isInCheck() const noexcept {
      constexpr Side enemy = getOtherSide(ally);
      const Square kingSq = peekPiece(bitboards_[ally][King]);
      const Bitboard occup = getOccupancy(White) | getOccupancy(Black);
      return ((getAttack<Pawn, ally>(kingSq) & bitboards_[enemy][Pawn]) ||
              (getAttack<Knight>(kingSq) & bitboards_[enemy][Knight]) ||
              (getAttack<Bishop>(kingSq, occup) & (bitboards_[enemy][Bishop] | bitboards_[enemy][Queen])) ||
              (getAttack<Rook>(kingSq, occup) & (bitboards_[enemy][Rook] | bitboards_[enemy][Queen])));
    }

    template <Side ally, bool violentOnly = false>
    constexpr void generateMoves(MoveList& moves) const noexcept {
      constexpr Side enemy = getOtherSide(ally);
      const Square kingSq = peekPiece(bitboards_[ally][King]);
      const std::array<Bitboard, kSideSize> occupancy = {getOccupancy(White), getOccupancy(Black)};
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
        constexpr Bitboard promoRank = (ally == White ? kRank8Mask : kRank1Mask);

        // Left capture
        const Bitboard leftCaptureDbb = (ally == White ? shiftUpLeft(bitboards_[ally][Pawn]) : shiftDownLeft(bitboards_[ally][Pawn])) & occupancy[enemy] & evasionMask;
        const auto getLeftSrce = [](Square dest) constexpr noexcept { return ally == White ? squareDownRight(dest) : squareUpRight(dest); };
        generatePawnMoves<false, true>(moves, leftCaptureDbb & ~promoRank, kingSq, pinnedMask, getLeftSrce);
        generatePawnMoves<true, true>(moves, leftCaptureDbb & promoRank, kingSq, pinnedMask, getLeftSrce);

        // Right capture
        const Bitboard rightCaptureDbb = (ally == White ? shiftUpRight(bitboards_[ally][Pawn]) : shiftDownRight(bitboards_[ally][Pawn])) & occupancy[enemy] & evasionMask;
        const auto getRightSrce = [](Square dest) constexpr noexcept { return ally == White ? squareDownLeft(dest) : squareUpLeft(dest); };
        generatePawnMoves<false, true>(moves, rightCaptureDbb & ~promoRank, kingSq, pinnedMask, getRightSrce);
        generatePawnMoves<true, true>(moves, rightCaptureDbb & promoRank, kingSq, pinnedMask, getRightSrce);

        // Push Forward
        const Bitboard singlePushBB = (ally == White ? shiftUp(bitboards_[ally][Pawn]) : shiftDown(bitboards_[ally][Pawn])) & ~bothOccupancy;
        const Bitboard pushDbb = singlePushBB & evasionMask;
        const auto getPushSrce = [](Square dest) constexpr noexcept { return ally == White ? squareDown(dest) : squareUp(dest); };
        if constexpr (!violentOnly) {
          generatePawnMoves<false, false>(moves, pushDbb & ~promoRank, kingSq, pinnedMask, getPushSrce);
        }
        generatePawnMoves<true, false>(moves, pushDbb & promoRank, kingSq, pinnedMask, getPushSrce);

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
      if constexpr (violentOnly) {
        for (Bitboard bb = getAttack<King>(kingSq) & occupancy[enemy] & ~attackedMask;
             bb;
             bb = popPiece(bb)) {
          const Square dest = peekPiece(bb);
          moves.push(Move(kingSq, dest, King, NoPiece, Move::kCaptureFlag));
        }
      } else {
        for (Bitboard bb = getAttack<King>(kingSq) & ~occupancy[ally] & ~attackedMask;
             bb;
             bb = popPiece(bb)) {
          const Square dest = peekPiece(bb);
          if (isSquareSet(occupancy[enemy], dest)) {
            moves.push(Move(kingSq, dest, King, NoPiece, Move::kCaptureFlag));
          } else {
            moves.push(Move(kingSq, dest, King));
          }
        }
        addCastlingMove<ally, true>(moves, bothOccupancy, attackedMask);
        addCastlingMove<ally, false>(moves, bothOccupancy, attackedMask);
      }
    }


    struct NoOpMoveCallback {
      template <bool>
      constexpr void markPiece(Side, Piece, Square) const noexcept {}
    };

    // Save only the irreversible state; piece moves can be reversed from Move.
    template <Side ally, bool isNullMove = false, typename MoveCallback = NoOpMoveCallback>
    constexpr MoveUndo makeMove([[maybe_unused]] Move move = Move{},
                                [[maybe_unused]] MoveCallback&& moveCallback = {}) noexcept {
      assert(side_ == ally);
      constexpr Side enemy = getOtherSide(ally);
      MoveUndo undo{
        .capturedPiece = NoPiece,
        .enpassant = enpassant_,
        .castlePermission = castlePermission_,
        .halfmove = halfmove_,
        .fullmove = fullmove_,
        .hash = zobrist_.hash(),
      };

      if constexpr (!isNullMove) {
        const Square srce = move.from();
        const Square dest = move.to();
        const Piece movedPiece = move.getMovedPiece();
        const Piece promotion = move.getPromotedPieceType();

        if (move.isEnpassant()) {
          // Perform enpassant.
          const Square capturedSq = (ally == White ? squareDown(dest) : squareUp(dest));
          bitboards_[enemy][Pawn] = unsetSquare(bitboards_[enemy][Pawn], capturedSq);
          undo.capturedPiece = Pawn;
          zobrist_.markPiece(enemy, Pawn, capturedSq);
          moveCallback.template markPiece<false>(enemy, Pawn, capturedSq);

        } else if (move.isCapture()) {
          // Perform capturing.
          const Piece capturedPiece = getPieceAt<enemy>(dest);
          assert(capturedPiece != NoPiece);
          bitboards_[enemy][capturedPiece] = unsetSquare(bitboards_[enemy][capturedPiece], dest);
          undo.capturedPiece = capturedPiece;
          zobrist_.markPiece(enemy, capturedPiece, dest);
          moveCallback.template markPiece<false>(enemy, capturedPiece, dest);
        }

        // Move the piece.
        bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], srce, dest);
        zobrist_.markPiece(ally, movedPiece, srce);
        zobrist_.markPiece(ally, movedPiece, dest);
        moveCallback.template markPiece<false>(ally, movedPiece, srce);
        if (promotion == NoPiece) {
          moveCallback.template markPiece<true>(ally, movedPiece, dest);
        }

        if (promotion != NoPiece) {
          // Perform promotion.
          bitboards_[ally][Pawn] = unsetSquare(bitboards_[ally][Pawn], dest);
          bitboards_[ally][promotion] = setSquare(bitboards_[ally][promotion], dest);
          zobrist_.markPiece(ally, movedPiece, dest);
          zobrist_.markPiece(ally, promotion, dest);
          moveCallback.template markPiece<true>(ally, promotion, dest);
        }
        if (move.isCastling()) {
          // Perform castling.
          const Square rookFrom = dest > srce ? (ally == White ? H1 : H8) : (ally == White ? A1 : A8);
          const Square rookTo = dest > srce ? dest - 1 : dest + 1;
          bitboards_[ally][Rook] = moveSquare(bitboards_[ally][Rook], rookFrom, rookTo);
          zobrist_.markPiece(ally, Rook, rookFrom);
          zobrist_.markPiece(ally, Rook, rookTo);
          moveCallback.template markPiece<false>(ally, Rook, rookFrom);
          moveCallback.template markPiece<true>(ally, Rook, rookTo);
        }

        // Update castle permission.
        zobrist_.markCastle(castlePermission_);
        castlePermission_ &= kCastlePermissionMask[srce] & kCastlePermissionMask[dest];
        zobrist_.markCastle(castlePermission_);

        // Update enpassant square.
        zobrist_.markEnpassant(enpassant_);
        enpassant_ = move.isDoublePush() ? (ally == White ? squareUp(srce) : squareDown(srce)) : NoSquare;
        zobrist_.markEnpassant(enpassant_);

        // Update half move.
        halfmove_ = (movedPiece == Pawn || undo.capturedPiece != NoPiece) ? 0 : halfmove_ + 1;
      } else {
        // Null move: clear enpassant square and reset halfmove clock.
        zobrist_.markEnpassant(enpassant_);
        enpassant_ = NoSquare;
        zobrist_.markEnpassant(enpassant_);
        halfmove_ = 0;
      }

      if constexpr (ally == Black) {
        ++fullmove_;
      }

      // Change side.
      side_ = enemy;
      zobrist_.markSide();

      return undo;
    }

    template <Side ally, bool isNullMove = false, typename MoveCallback = NoOpMoveCallback>
    constexpr void unmakeMove([[maybe_unused]] Move move,
                              const MoveUndo& undo,
                              [[maybe_unused]] MoveCallback&& moveCallback = {}) noexcept {
      assert(side_ == getOtherSide(ally));
      if constexpr (!isNullMove) {
        constexpr Side enemy = getOtherSide(ally);
        const Square srce = move.from();
        const Square dest = move.to();

        if (move.isCastling()) {
          const Square rookFrom = dest > srce ? (ally == White ? H1 : H8) : (ally == White ? A1 : A8);
          const Square rookTo = dest > srce ? dest - 1 : dest + 1;
          bitboards_[ally][Rook] = moveSquare(bitboards_[ally][Rook], rookTo, rookFrom);
          moveCallback.template markPiece<false>(ally, Rook, rookTo);
          moveCallback.template markPiece<true>(ally, Rook, rookFrom);
        }
        if (const Piece promotion = move.getPromotedPieceType(); promotion != NoPiece) {
          bitboards_[ally][promotion] = unsetSquare(bitboards_[ally][promotion], dest);
          bitboards_[ally][Pawn] = setSquare(bitboards_[ally][Pawn], srce);
          moveCallback.template markPiece<false>(ally, promotion, dest);
          moveCallback.template markPiece<true>(ally, Pawn, srce);
        } else {
          const Piece movedPiece = move.getMovedPiece();
          bitboards_[ally][movedPiece] = moveSquare(bitboards_[ally][movedPiece], dest, srce);
          moveCallback.template markPiece<false>(ally, movedPiece, dest);
          moveCallback.template markPiece<true>(ally, movedPiece, srce);
        }
        if (undo.capturedPiece != NoPiece) {
          const Square capturedSq = move.isEnpassant() ? (ally == White ? squareDown(dest) : squareUp(dest)) : dest;
          bitboards_[enemy][undo.capturedPiece] = setSquare(bitboards_[enemy][undo.capturedPiece], capturedSq);
          moveCallback.template markPiece<true>(enemy, undo.capturedPiece, capturedSq);
        }
        castlePermission_ = undo.castlePermission;
      }
      enpassant_ = undo.enpassant;
      halfmove_ = undo.halfmove;
      fullmove_ = undo.fullmove;
      side_ = ally;
      zobrist_.set(undo.hash);
    }

    template <Side ally, bool isNullMove, typename MoveCallback = NoOpMoveCallback>
      requires isNullMove
    constexpr void unmakeMove(const MoveUndo& undo, MoveCallback&& moveCallback = {}) noexcept {
      unmakeMove<ally, true>(Move{}, undo, std::forward<MoveCallback>(moveCallback));
    }
  };
  static_assert(std::is_trivially_copyable_v<BoardState>, "BoardState must remain cheap to copy");
}
