#pragma once
#include "attack.h"
#include "move.h"
#include <array>
#include <set>

///////////////////////////////////////////////////////
//                 CHESS BOARD STATE
///////////////////////////////////////////////////////
class BoardState {
public:
  std::array<std::array<Bitboard, kPieceSize>, kColorSize> bitboards_;
  std::array<Bitboard, kColorSize> occupancy_;
  Bitboard bothOccupancy_;
  Bitboard castlePermission_;
  Square enpassant_;
  Color color_;
  int32_t halfmove_;
  int32_t fullmove_;

  // Return our pawns' attack bitboard.
  template <Color our>
  constexpr Bitboard getPawnAttack() const {
    if constexpr (our == kWhite) {
      return shiftUpLeft(bitboards_[our][kPawn]) | shiftUpRight(bitboards_[our][kPawn]);
    } else {
      return shiftDownLeft(bitboards_[our][kPawn]) | shiftDownRight(bitboards_[our][kPawn]);
    }
  }

  // Return a bitboard containing squares attacked by their pieces.
  template <Color our>
  constexpr Bitboard getAttacked() const {
    constexpr Color their = getOtherColor(our);

    // If king blocks the attack ray, then it may incorrectly move backward illegally.
    // Consider the move: r...K... -> r....K..
    Bitboard occupancy = bothOccupancy_ & ~bitboards_[our][kKing];
    Bitboard attacked = getPawnAttack<their>();

    {
      Square srce = getFirstPieceSquare(bitboards_[their][kKing]);
      attacked |= kAttackTable<kKing>[srce];
    }
    for (Bitboard bb = bitboards_[their][kKnight]; bb; bb = removeFirstPiece(bb)) {
      Square srce = getFirstPieceSquare(bb);
      attacked |= kAttackTable<kKnight>[srce];
    }
    for (Bitboard bb = bitboards_[their][kBishop]; bb; bb = removeFirstPiece(bb)) {
      Square srce = getFirstPieceSquare(bb);
      attacked |= SliderTable::getAttack<kBishop>(srce, occupancy);
    }
    for (Bitboard bb = bitboards_[their][kRook]; bb; bb = removeFirstPiece(bb)) {
      Square srce = getFirstPieceSquare(bb);
      attacked |= SliderTable::getAttack<kRook>(srce, occupancy);
    }
    for (Bitboard bb = bitboards_[their][kQueen]; bb; bb = removeFirstPiece(bb)) {
      Square srce = getFirstPieceSquare(bb);
      attacked |= SliderTable::getAttack<kQueen>(srce, occupancy);
    }

    return attacked;
  }

  // Return a bitboard containing their pieces that are giving checks.
  template <Color our>
  constexpr Bitboard getCheckers() const {
    constexpr Color their = getOtherColor(our);
    Square kingSq = getFirstPieceSquare(bitboards_[our][kKing]);

    // Ignore enemy king because king can't check king.
    return (kAttackTable<kPawn>[our][kingSq] & bitboards_[their][kPawn]) |
           (kAttackTable<kKnight>[kingSq] & bitboards_[their][kKnight]) |
           (SliderTable::getAttack<kBishop>(kingSq, bothOccupancy_) & (bitboards_[their][kBishop] | bitboards_[their][kQueen])) |
           (SliderTable::getAttack<kRook>(kingSq, bothOccupancy_) & (bitboards_[their][kRook] | bitboards_[their][kQueen]));
  }

  // Return a bitboard containing our pieces that are pinned.
  template <Color our>
  constexpr Bitboard getPinned() const {
    constexpr Color their = getOtherColor(our);
    Square kingSq = getFirstPieceSquare(bitboards_[our][kKing]);

    // Get the enemy sliders squares, then check if any ally piece is blocking the attack ray.
    Bitboard pinned{};
    Bitboard sliders = (SliderTable::getAttack<kBishop>(kingSq, occupancy_[their]) & (bitboards_[their][kBishop] | bitboards_[their][kQueen])) |
                            (SliderTable::getAttack<kRook>(kingSq, occupancy_[their]) & (bitboards_[their][kRook] | bitboards_[their][kQueen]));
    for (; sliders; sliders = removeFirstPiece(sliders)) {
      Square sliderSquare = getFirstPieceSquare(sliders);
      Bitboard blockers = kSquareBetweenMasks[kingSq][sliderSquare] & occupancy_[our];
      if (countPiece(blockers) == 1) {
        pinned |= blockers; // Does NOT handle enpassant edge case.
      }
    }
    return pinned;
  }

  template <Color our, Piece piece>
  constexpr void getLegalPieceMove(MoveList& moveList, Square kingSq, Bitboard blockOrCaptureMask, Bitboard pinned) {
    Bitboard sbb = bitboards_[our][piece];
    if constexpr (piece == kKnight) {
      sbb &= ~pinned; // Pinned knight can never move.
    }

    for (; sbb; sbb = removeFirstPiece(sbb)) {
      Square srce = getFirstPieceSquare(sbb);

      // Don't attack our pieces, and must block or capture checker if there's any.
      Bitboard dbb = ~occupancy_[our] & blockOrCaptureMask;

      // Restirct the piece movement in the pinned direction.
      if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
        if (isSquareSet(pinned, srce)) {
          dbb &= kLineOfSightMasks[kingSq][srce];
        }
      }

      // Generate actual moves.
      if constexpr (piece == kKnight) {
        dbb &= kAttackTable<kKnight>[srce];
      } else {
        dbb &= SliderTable::getAttack<piece>(srce, bothOccupancy_);
      }

      for (; dbb; dbb = removeFirstPiece(dbb)) {
        Square dest = getFirstPieceSquare(dbb);
        moveList.push(Move(srce, dest));
      }
    }
  }

  template <Color our>
  constexpr void getLegalMove(MoveList& moveList) const {
    const Square kingSq = getFirstPieceSquare(bitboards_[our][kKing]);
    const Bitboard pinned = getPinned<our>();

    // Our piece must block the check or capture the checker.
    // Add the attack ray as the constraints.
    Bitboard blockOrCaptureMask = ~Bitboard{};
    for (Bitboard checkers = getCheckers<our>(); checkers; checkers = removeFirstPiece(checkers)) {
      Square checkerSq = getFirstPieceSquare(checkers);
      blockOrCaptureMask &= setSquare(kSquareBetweenMasks[kingSq][checkerSq], checkerSq);
    }

    // Knight, Bishop, Rook, Queen Moves
    getLegalPieceMove<our, kKnight>(moveList, kingSq, blockOrCaptureMask, pinned);
    getLegalPieceMove<our, kBishop>(moveList, kingSq, blockOrCaptureMask, pinned);
    getLegalPieceMove<our, kRook>(moveList, kingSq, blockOrCaptureMask, pinned);
    getLegalPieceMove<our, kQueen>(moveList, kingSq, blockOrCaptureMask, pinned);

    
    // Pawn Moves
    {
      for (Bitboard sbb = bitboards_[our][kPawn]; sbb; sbb = removeFirstPiece(sbb)) {
        const Square srce = getFirstPieceSquare(sbb);
        const Bitboard srceBB = toBitboard(srce);

        // Attack and Push.
        const Bitboard attackBB = kAttackTable<kPawn>[our][srce] & occupancy_[their];
        const Bitboard singlePushBB = (our == kWhite ? shiftUp(srceBB) : shiftDown(srceBB)) & ~bothOccupancy_;
        const Bitboard doublePushBB = (our == kWhite ? shiftUp(singlePushBB) & kRank4Mask : shiftDown(singlePushBB) & kRank5Mask) & ~bothOccupancy_;
        Bitboard dbb = (attackBB | singlePushBB | doublePushBB) & blockOrCaptureMask;

        // Apply pins.
        if (isSquareSet(pinned, srce)) {
          dbb &= kLineOfSightMasks[kingSq][srce];
        }

        for (; dbb; dbb = removeFirstPiece(dbb)) {
          Square dest = getFirstPieceSquare(dbb);
          if (dest == kPromotionRank[our]) {
            moveList.push(Move(srce, dest, kPawn, kKnight));
            moveList.push(Move(srce, dest, kPawn, kBishop));
            moveList.push(Move(srce, dest, kPawn, kRook));
            moveList.push(Move(srce, dest, kPawn, kQueen));
          } else {
            moveList.(Move(srce, dest));
          }
        }
      }

      // Enpassant
    }


    // King Moves
    {
      const Bitboard attacked = getAttacked<our>();

      // King Walk
      for (Bitboard bb = kAttackTable<kKing>[kingSq] & ~occupancy_[our] & ~attacked;
           bb;
           bb = removeFirstPiece(bb)) {
        moveList.push(Move(kingSq, getFirstPieceSquare(bb)));
      }

      // King Castling
      if ((castlePermission_ & kKingCastlePermission[our]) == kKingCastlePermission[our] &&  // Check castle permission
          (bothOccupancy_ & kKingCastleOccupancy[our]) == 0 &&                                    // Check castle blocker
          (attacked & kKingCastleSafety[our]) == 0) {                                             // Check castle attacked squares
        if constexpr (our == kWhite) {
          moveList.push(Move(E1, G1));
        } else {
          moveList.push(Move(E8, G8));
        }
      }

      // Queen Castling
      if ((castlePermission_ & kQueenCastlePermission[our]) == kQueenCastlePermission[our] && // Check castle permission
          (bothOccupancy_ & kQueenCastleOccupancy[our]) == 0 &&                                    // Check castle blocker
          (attacked & kQueenCastleSafety[our]) == 0) {                                             // Check castle attacked squares
        if constexpr (our == kWhite) {
          moveList.push(Move(E1, C1));
        } else {
          moveList.push(Move(E8, C8));
        }
      }
    }
  }























  ///////////////////////////////////////////////////////
  //             PSEUDO MOVE GENERATOR
  ///////////////////////////////////////////////////////

  // Return true if the square is attacked by the attacker.
  template <Color our>
  constexpr bool isAttacked(Square square) const {
    constexpr Color their = getOtherColor(our);

    if ((kAttackTable<kPawn>[their][square] & bitboards_[our][kPawn]) |
        (kAttackTable<kKnight>[square] & bitboards_[our][kKnight]) |
        (kAttackTable<kKing>[square] & bitboards_[our][kKing])) {
      return true;
    }

    const Bitboard diagonalAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    if (diagonalAttack & (bitboards_[our][kBishop] | bitboards_[our][kQueen])) {
      return true;
    }

    const Bitboard lineAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    return lineAttack & (bitboards_[our][kRook] | bitboards_[our][kQueen]);
  }

  template <Color our, Piece piece>
  constexpr void getPseudoPieceMove(MoveList& moveList) const {
    constexpr Color their = getOtherColor(our);

    for (Bitboard sbb = bitboards_[our][piece];
         sbb;
         sbb = removeFirstPiece(sbb)) {
      Square source = getFirstPieceSquare(sbb);
      Bitboard attackBB;
      if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
        attackBB = SliderTable::getAttack<piece>(source, bothOccupancy_);
      } else {
        attackBB = kAttackTable<piece>[source];
      }

      // Non-capture
      for (Bitboard dbb = attackBB & ~occupancy_[our];
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        uint32_t flag = isSquareSet(occupancy_[their], destination) ? Move::kCaptureFlag : 0;
        moveList.push(Move(source, destination, piece, 0, flag));
      }
    }
  }

  // TODO: consider put NO_SQUARE in kPawnAttackTable and give it an empty mask
  // TODO: separate capture move, promotion, and quiet move
  // TODO: generate all pawn moves at once
  template <Color our>
  constexpr void getPseudoMove(MoveList& moveList) const {
    constexpr Color their = getOtherColor(our);

    // Pawn
    {
      const Bitboard pushOnceBB = (our == kWhite ? shiftUp(bitboards_[our][kPawn]) : shiftDown(bitboards_[our][kPawn])) & ~bothOccupancy_;

      // Push once
      for (Bitboard dbb = pushOnceBB;
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (our == kWhite ? destination + kSideSize : destination - kSideSize);
        
        if (getSquareRank(destination) != kPromotionRank[our]) {
          moveList.push(Move(source, destination, kPawn));
        } else {
          moveList.push(Move(source, destination, kPawn, kKnight));
          moveList.push(Move(source, destination, kPawn, kBishop));
          moveList.push(Move(source, destination, kPawn, kRook));
          moveList.push(Move(source, destination, kPawn, kQueen));
        }
      }

      // Push twice
      for (Bitboard dbb = (our == kWhite ? shiftUp(pushOnceBB) & kRank4Mask : shiftDown(pushOnceBB) & kRank5Mask) & ~bothOccupancy_;
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (our == kWhite ? destination + 2 * kSideSize : destination - 2 * kSideSize);
        moveList.push(Move(source, destination, kPawn, 0, Move::kDoublePushFlag));
      }

      // Capture and promotion
      for (Bitboard sbb = bitboards_[our][kPawn];
           sbb;
           sbb = removeFirstPiece(sbb)) {
        Square source = getFirstPieceSquare(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[our][source] & occupancy_[their]; dbb; dbb = removeFirstPiece(dbb)) {
          Square destination = getFirstPieceSquare(dbb);
          if (getSquareRank(destination) != kPromotionRank[our]) {
            moveList.push(Move(source, destination, kPawn, 0, Move::kCaptureFlag));
          } else {
            moveList.push(Move(source, destination, kPawn, kKnight, Move::kCaptureFlag));
            moveList.push(Move(source, destination, kPawn, kBishop, Move::kCaptureFlag));
            moveList.push(Move(source, destination, kPawn, kRook, Move::kCaptureFlag));
            moveList.push(Move(source, destination, kPawn, kQueen, Move::kCaptureFlag));
          }
        }
      }

      // Enpassant
      if (enpassant_ != NO_SQUARE) {
        for (Bitboard enpassantBB = kAttackTable<kPawn>[their][enpassant_] & bitboards_[our][kPawn];
             enpassantBB;
             enpassantBB = removeFirstPiece(enpassantBB)) {
          Square source = getFirstPieceSquare(enpassantBB);
          moveList.push(Move(source, enpassant_, kPawn, 0, Move::kCaptureFlag|Move::kEnpassantFlag));
        }
      }
    }

    // Piece moves
    getPseudoPieceMove<our, kKnight>(moveList);
    getPseudoPieceMove<our, kKing>(moveList);
    getPseudoPieceMove<our, kBishop>(moveList);
    getPseudoPieceMove<our, kRook>(moveList);
    getPseudoPieceMove<our, kQueen>(moveList);

    // Castle
    {
      constexpr Square kingSourceSquare = (our == kWhite ? E1 : E8);
      constexpr Square kingSideMiddleSquare = (our == kWhite ? F1 : F8);
      constexpr Square kingSideDestinationSquare = (our == kWhite ? G1 : G8);
      constexpr Square queenSideMiddleSquare = (our == kWhite ? D1 : D8);
      constexpr Square queenSideDestinationSquare = (our == kWhite ? C1 : C8);

      // Check castle permission, blocker occupancy, and castle square safety.
      // Must check the king and the adjacent square to prevent illegal castle that makes king safe.
      // No need to check the king square after castle, because it will get verified by the legal move generator later.
      if ((castlePermission_ & kKingCastlePermission[our]) == kKingCastlePermission[our] && 
          (kKingCastleOccupancy[our] & bothOccupancy_) == 0 &&
          !isAttacked<their>(kingSourceSquare) && !isAttacked<their>(kingSideMiddleSquare)) {
        moveList.push(Move(kingSourceSquare, kingSideDestinationSquare, kKing, 0, Move::kCastlingFlag));
      }

      if ((castlePermission_ & kQueenCastlePermission[our]) == kQueenCastlePermission[our] && 
          (kQueenCastleOccupancy[our] & bothOccupancy_) == 0 &&
          !isAttacked<their>(kingSourceSquare) && !isAttacked<their>(queenSideMiddleSquare)) {
        moveList.push(Move(kingSourceSquare, queenSideDestinationSquare, kKing, 0, Move::kCastlingFlag));
      }
    }

    // Check duplicated moves.
    assert(std::set<Move>(moveList.begin(), moveList.end()).size() == moveList.size());
  }
  
  static BoardState fromFEN(const std::string& fen);

  friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
};
static_assert(std::is_trivial_v<BoardState>, "BoardState is not POD type, may affect performance");


///////////////////////////////////////////////////////
//                 CHESS BOARD
///////////////////////////////////////////////////////
class Board {
protected:
  BoardState currentState_;

  // Play the move and change the current state regardless its legality.
  // Return true if the move is legal, false otherwise.
  template <Color our>
  bool tryMakeMove(Move move) {
    constexpr Color their = getOtherColor(our);
    const Square source = move.getSourceSquare();
    const Square destination = move.getDestinationSquare();
    const Piece movedPiece = move.getMovedPiece();

    // Move the piece.
    currentState_.bitboards_[our][movedPiece] = moveSquare(currentState_.bitboards_[our][movedPiece], source, destination);
    currentState_.occupancy_[our] = moveSquare(currentState_.occupancy_[our], source, destination);
    currentState_.enpassant_ = NO_SQUARE;
    ++currentState_.halfmove_;
    if constexpr (our == kBlack) {
      ++currentState_.fullmove_;
    }

    if (move.isCaptured()) {
      currentState_.bitboards_[their][kPawn] = unsetSquare(currentState_.bitboards_[their][kPawn], destination);
      currentState_.bitboards_[their][kKnight] = unsetSquare(currentState_.bitboards_[their][kKnight], destination);
      currentState_.bitboards_[their][kBishop] = unsetSquare(currentState_.bitboards_[their][kBishop], destination);
      currentState_.bitboards_[their][kRook] = unsetSquare(currentState_.bitboards_[their][kRook], destination);
      currentState_.bitboards_[their][kQueen] = unsetSquare(currentState_.bitboards_[their][kQueen], destination);
      currentState_.occupancy_[their] = unsetSquare(currentState_.occupancy_[their], destination); // Does not affect enpassant
      currentState_.halfmove_ = 0;
    }

    if (movedPiece == kPawn) {
      currentState_.halfmove_ = 0;
    }

    currentState_.halfmove_ = 0;
    if (move.isDoublePush()) {
      currentState_.enpassant_ = (our == kWhite ? destination + 8 : destination - 8);

    } else if (move.isEnpassant()) {
      Square enpassantPawnSquare = (our == kWhite ? destination + 8 : destination - 8);
      currentState_.bitboards_[their][kPawn] = unsetSquare(currentState_.bitboards_[their][kPawn], enpassantPawnSquare);
      currentState_.occupancy_[their] = unsetSquare(currentState_.occupancy_[their], enpassantPawnSquare);

    } else if (Piece promotedPiece = move.getPromotedPiece(); promotedPiece) {
      currentState_.bitboards_[our][movedPiece] = unsetSquare(currentState_.bitboards_[our][movedPiece], destination);
      currentState_.bitboards_[our][promotedPiece] = setSquare(currentState_.bitboards_[our][promotedPiece], destination);

    } else if (move.isCastling()) {
      Square rookSource;
      Square rookDestination;
      if constexpr (our == kWhite) {
        if (destination == G1) {
          rookSource = H1;
          rookDestination = F1;
        } else {
          rookSource = A1;
          rookDestination = D1;
        }
      } else {
        if (destination == G8) {
          rookSource = H8;
          rookDestination = F8;
        } else {
          rookSource = A8;
          rookDestination = D8;
        }
      }

      currentState_.bitboards_[our][kRook] = moveSquare(currentState_.bitboards_[our][kRook], rookSource, rookDestination);
      currentState_.occupancy_[our] = moveSquare(currentState_.occupancy_[our], rookSource, rookDestination);
    }

    // Update castle permission and occupancy
    currentState_.castlePermission_ = unsetSquare(unsetSquare(currentState_.castlePermission_, source), destination);
    currentState_.bothOccupancy_ = currentState_.occupancy_[kWhite] | currentState_.occupancy_[kBlack];

#ifdef _DEBUG
    // Check occupancy consistency.
    for (Color color : {kWhite, kBlack}) {
      Bitboard occupancy = 0;
      for (Bitboard bb : currentState_.bitboards_[color]) {
        occupancy |= bb;
      }
      if (occupancy != currentState_.occupancy_[color]) {
        std::cout << move.toString() << '\n';
        printBitboard(occupancy);
        printBitboard(currentState_.occupancy_[color]);
        assert(false && "occupancy inconsistent");
      }
    }
#endif

    currentState_.color_ = their;

    // Check if the move is legal
    const Square kingSquare = getFirstPieceSquare(currentState_.bitboards_[our][kKing]);
    return !currentState_.isAttacked<their>(kingSquare);
  }

public:
  explicit Board(const std::string& fen) {
    setFEN(fen);
  }

  void setFEN(const std::string& fen) {
    currentState_ = BoardState::fromFEN(fen);
  }

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
