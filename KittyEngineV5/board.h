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


  


























  ///////////////////////////////////////////////////////
  //             PSEUDO MOVE GENERATOR
  ///////////////////////////////////////////////////////

  // Return true if the square is attacked by the attacker.
  template <Color us>
  constexpr bool isAttacked(Square square) const {
    constexpr Color them = getOtherColor(us);

    if ((kAttackTable<kPawn>[them][square] & bitboards_[us][kPawn]) |
        (kAttackTable<kKnight>[square] & bitboards_[us][kKnight]) |
        (kAttackTable<kKing>[square] & bitboards_[us][kKing])) {
      return true;
    }

    const Bitboard diagonalAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    if (diagonalAttack & (bitboards_[us][kBishop] | bitboards_[us][kQueen])) {
      return true;
    }

    const Bitboard lineAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    return lineAttack & (bitboards_[us][kRook] | bitboards_[us][kQueen]);
  }

  template <Color us, Piece piece>
  constexpr void getPseudoPieceMove(MoveList& moveList) const {
    constexpr Color them = getOtherColor(us);

    for (Bitboard sbb = bitboards_[us][piece];
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
      for (Bitboard dbb = attackBB & ~occupancy_[us];
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        uint32_t flag = isSquareSet(occupancy_[them], destination) ? Move::kCaptureFlag : 0;
        moveList.push(Move(source, destination, piece, 0, flag));
      }
    }
  }

  // TODO: consider put NO_SQUARE in kPawnAttackTable and give it an empty mask
  // TODO: separate capture move, promotion, and quiet move
  // TODO: generate all pawn moves at once
  template <Color us>
  constexpr void getPseudoMove(MoveList& moveList) const {
    constexpr Color them = getOtherColor(us);

    // Pawn
    {
      const Bitboard pushOnceBB = (us == kWhite ? shiftUp(bitboards_[us][kPawn]) : shiftDown(bitboards_[us][kPawn])) & ~bothOccupancy_;

      // Push once
      for (Bitboard dbb = pushOnceBB;
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (us == kWhite ? destination + kSideSize : destination - kSideSize);
        
        constexpr Square promotionRank = (us == kWhite ? 0 : 7);
        if (getSquareRank(destination) != promotionRank) {
          moveList.push(Move(source, destination, kPawn));
        } else {
          moveList.push(Move(source, destination, kPawn, kKnight));
          moveList.push(Move(source, destination, kPawn, kBishop));
          moveList.push(Move(source, destination, kPawn, kRook));
          moveList.push(Move(source, destination, kPawn, kQueen));
        }
      }

      // Push twice
      for (Bitboard dbb = (us == kWhite ? shiftUp(pushOnceBB) & kRank4Mask : shiftDown(pushOnceBB) & kRank5Mask) & ~bothOccupancy_;
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (us == kWhite ? destination + 2 * kSideSize : destination - 2 * kSideSize);
        moveList.push(Move(source, destination, kPawn, 0, Move::kDoublePushFlag));
      }

      // Capture and promotion
      for (Bitboard sbb = bitboards_[us][kPawn];
           sbb;
           sbb = removeFirstPiece(sbb)) {
        Square source = getFirstPieceSquare(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[us][source] & occupancy_[them]; dbb; dbb = removeFirstPiece(dbb)) {
          Square destination = getFirstPieceSquare(dbb);
          constexpr Square promotionRank = (us == kWhite ? 0 : 7);
          if (getSquareRank(destination) != promotionRank) {
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
        for (Bitboard enpassantBB = kAttackTable<kPawn>[them][enpassant_] & bitboards_[us][kPawn];
             enpassantBB;
             enpassantBB = removeFirstPiece(enpassantBB)) {
          Square source = getFirstPieceSquare(enpassantBB);
          moveList.push(Move(source, enpassant_, kPawn, 0, Move::kCaptureFlag|Move::kEnpassantFlag));
        }
      }
    }

    // Piece moves
    getPseudoPieceMove<us, kKnight>(moveList);
    getPseudoPieceMove<us, kKing>(moveList);
    getPseudoPieceMove<us, kBishop>(moveList);
    getPseudoPieceMove<us, kRook>(moveList);
    getPseudoPieceMove<us, kQueen>(moveList);

    // Castle
    {
      constexpr Bitboard kingCastlePermission = (us == kWhite ? kWhiteKingCastlePermission : kBlackKingCastlePermission);
      constexpr Bitboard queenCastlePermission = (us == kWhite ? kWhiteQueenCastlePermission : kBlackQueenCastlePermission);
      constexpr Bitboard kingCastleOccupancy = (us == kWhite ? 0x6000000000000000ull : 0x60ul);
      constexpr Bitboard queenCastleOccupancy = (us == kWhite ? 0xE00000000000000ull : 0xEull);
      constexpr Square kingSourceSquare = (us == kWhite ? E1 : E8);
      constexpr Square kingSideMiddleSquare = (us == kWhite ? F1 : F8);
      constexpr Square kingSideDestinationSquare = (us == kWhite ? G1 : G8);
      constexpr Square queenSideMiddleSquare = (us == kWhite ? D1 : D8);
      constexpr Square queenSideDestinationSquare = (us == kWhite ? C1 : C8);

      // Check castle permission, blocker occupancy, and castle square safety.
      // Must check the king and the adjacent square to prevent illegal castle that makes king safe.
      // No need to check the king square after castle, because it will get verified by the legal move generator later.
      if (((castlePermission_ & kingCastlePermission) == kingCastlePermission) && (kingCastleOccupancy & bothOccupancy_) == 0 &&
          !isAttacked<them>(kingSourceSquare) && !isAttacked<them>(kingSideMiddleSquare)) {
        moveList.push(Move(kingSourceSquare, kingSideDestinationSquare, kKing, 0, Move::kCastlingFlag));
      }

      if (((castlePermission_ & queenCastlePermission) == queenCastlePermission) && (queenCastleOccupancy & bothOccupancy_) == 0 &&
          !isAttacked<them>(kingSourceSquare) && !isAttacked<them>(queenSideMiddleSquare)) {
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
  template <Color us>
  bool tryMakeMove(Move move) {
    constexpr Color them = getOtherColor(us);
    const Square source = move.getSourceSquare();
    const Square destination = move.getDestinationSquare();
    const Piece movedPiece = move.getMovedPiece();

    // Move the piece.
    currentState_.bitboards_[us][movedPiece] = moveSquare(currentState_.bitboards_[us][movedPiece], source, destination);
    currentState_.occupancy_[us] = moveSquare(currentState_.occupancy_[us], source, destination);
    currentState_.enpassant_ = NO_SQUARE;
    ++currentState_.halfmove_;
    if constexpr (us == kBlack) {
      ++currentState_.fullmove_;
    }

    if (move.isCaptured()) {
      currentState_.bitboards_[them][kPawn] = unsetSquare(currentState_.bitboards_[them][kPawn], destination);
      currentState_.bitboards_[them][kKnight] = unsetSquare(currentState_.bitboards_[them][kKnight], destination);
      currentState_.bitboards_[them][kBishop] = unsetSquare(currentState_.bitboards_[them][kBishop], destination);
      currentState_.bitboards_[them][kRook] = unsetSquare(currentState_.bitboards_[them][kRook], destination);
      currentState_.bitboards_[them][kQueen] = unsetSquare(currentState_.bitboards_[them][kQueen], destination);
      currentState_.occupancy_[them] = unsetSquare(currentState_.occupancy_[them], destination); // Does not affect enpassant
      currentState_.halfmove_ = 0;
    }

    if (movedPiece == kPawn) {
      currentState_.halfmove_ = 0;
    }

    currentState_.halfmove_ = 0;
    if (move.isDoublePush()) {
      currentState_.enpassant_ = (us == kWhite ? destination + 8 : destination - 8);

    } else if (move.isEnpassant()) {
      Square enpassantPawnSquare = (us == kWhite ? destination + 8 : destination - 8);
      currentState_.bitboards_[them][kPawn] = unsetSquare(currentState_.bitboards_[them][kPawn], enpassantPawnSquare);
      currentState_.occupancy_[them] = unsetSquare(currentState_.occupancy_[them], enpassantPawnSquare);

    } else if (Piece promotedPiece = move.getPromotedPiece(); promotedPiece) {
      currentState_.bitboards_[us][movedPiece] = unsetSquare(currentState_.bitboards_[us][movedPiece], destination);
      currentState_.bitboards_[us][promotedPiece] = setSquare(currentState_.bitboards_[us][promotedPiece], destination);

    } else if (move.isCastling()) {
      Square rookSource;
      Square rookDestination;
      if constexpr (us == kWhite) {
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

      currentState_.bitboards_[us][kRook] = moveSquare(currentState_.bitboards_[us][kRook], rookSource, rookDestination);
      currentState_.occupancy_[us] = moveSquare(currentState_.occupancy_[us], rookSource, rookDestination);
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

    currentState_.color_ = them;

    // Check if the move is legal
    const Square kingSquare = getFirstPieceSquare(currentState_.bitboards_[us][kKing]);
    return !currentState_.isAttacked<them>(kingSquare);
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
