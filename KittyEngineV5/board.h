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
  std::array<std::array<Bitboard, kPieceSize>, kTeamSize> bitboards_;
  std::array<Bitboard, kTeamSize> occupancy_;
  Bitboard bothOccupancy_;
  Bitboard castlePermission_;
  Square enpassant_;
  Team team_;
  int32_t halfmove_;
  int32_t fullmove_;

  // Return all the attackers that attack the square.
  template <Team attacker>
  constexpr Bitboard getAllAttackers(Square square) const {
    constexpr Team defender = getOtherTeam(attacker);
    const Bitboard bishopAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    const Bitboard rookAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    const Bitboard queenAttack = bishopAttack | rookAttack;

    // Simulate all attacks at the square. If you can attack the enemy, then they can attack you.
    return (kAttackTable<kPawn>[defender][square] & bitboards_[attacker][kPawn]) |
           (kAttackTable<kKnight>[square] & bitboards_[attacker][kKnight]) |
           (kAttackTable<kKing>[square] & bitboards_[attacker][kKing]) |
           (bishopAttack & bitboards_[attacker][kBishop]) | 
           (rookAttack & bitboards_[attacker][kRook]) | 
           (queenAttack & bitboards_[attacker][kQueen]);
  }

  // Return true if the square is attacked by the attacker.
  template <Team attacker>
  constexpr bool isAttacked(Square square) const {
    constexpr Team defender = getOtherTeam(attacker);

    if ((kAttackTable<kPawn>[defender][square] & bitboards_[attacker][kPawn]) |
        (kAttackTable<kKnight>[square] & bitboards_[attacker][kKnight]) |
        (kAttackTable<kKing>[square] & bitboards_[attacker][kKing])) {
      return true;
    }

    const Bitboard bishopAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    if (bishopAttack & bitboards_[attacker][kBishop]) {
      return true;
    }

    const Bitboard rookAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    if (rookAttack & bitboards_[attacker][kRook]) {
      return true;
    }

    const Bitboard queenAttack = bishopAttack | rookAttack;
    return queenAttack & bitboards_[attacker][kQueen];
  }

  template <Team attacker>
  constexpr void getLegalMove(MoveList& moveList) const {
    constexpr Team defender = getOtherTeam(attacker);
    const Square kingSquare = getFirstPieceSquare(bitboards_[attacker][kKing]);
    const Bitboard allAttacks = getAllAttackers<defender>(kingSquare);

    switch (countPiece(allAttacks)) {
    case 0: 
      // All moves except pinned pieces.

      // Identify the pinned pieces.

    case 1: 
      // Single Check: Capture the pieces, block the attack ray, or King evasion.

      // Identify the pinned pieces.


    case 2: {
      // Double Checks: king evasion.
      for (Bitboard dbb = kAttackTable<kKing>[kingSquare] & ~occupancy_[attacker]; // Empty and enemy occupancy.
            dbb;
            dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        if (!isAttacked<defender>(destination)) {
          uint32_t flag = (isSquareSet(occupancy_[defender], destination) ? Move::kCaptureFlag : 0);
          moveList.push(Move(kingSquare, destination, kKing, 0, flag));
        }
      }
      return;
    }
    }
  }





























  template <Team attacker, Piece piece>
  constexpr void getPseudoPieceMove(MoveList& moveList) const {
    constexpr Team defender = getOtherTeam(attacker);

    for (Bitboard sbb = bitboards_[attacker][piece];
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
      for (Bitboard dbb = attackBB & ~occupancy_[attacker];
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        uint32_t flag = isSquareSet(occupancy_[defender], destination) ? Move::kCaptureFlag : 0;
        moveList.push(Move(source, destination, piece, 0, flag));
      }

      //// Non-capture
      //for (Bitboard dbb = attackBB & ~bothOccupancy_;
      //     dbb;
      //     dbb = removeFirstPiece(dbb)) {
      //  Square destination = getFirstPieceSquare(dbb);
      //  moveList.push(Move(source, destination, piece));
      //}

      //// Capture
      //for (Bitboard dbb = attackBB & occupancy_[defender];
      //     dbb;
      //     dbb = removeFirstPiece(dbb)) {
      //  Square destination = getFirstPieceSquare(dbb);
      //  moveList.push(Move(source, destination, piece, 0, Move::kCaptureFlag));
      //}
    }
  }

  // TODO: consider put NO_SQUARE in kPawnAttackTable and give it an empty mask
  // TODO: separate capture move, promotion, and quiet move
  // TODO: generate all pawn moves at once
  template <Team attacker>
  constexpr void getPseudoMove(MoveList& moveList) const {
    constexpr Team defender = getOtherTeam(attacker);

    // Pawn
    {
      const Bitboard pushOnceBB = (attacker == kWhite ? shiftUp(bitboards_[attacker][kPawn]) : shiftDown(bitboards_[attacker][kPawn])) & ~bothOccupancy_;

      // Push once without promotion
      for (Bitboard dbb = pushOnceBB & (attacker == kWhite ? ~kRank8Mask : ~kRank1Mask);
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (attacker == kWhite ? destination + kSideSize : destination - kSideSize);
        moveList.push(Move(source, destination, kPawn));
      }

      // Push once with promotion
      for (Bitboard dbb = pushOnceBB & (attacker == kWhite ? kRank8Mask : kRank1Mask);
           dbb;
           dbb = removeFirstPiece(dbb)) {
        Square destination = getFirstPieceSquare(dbb);
        Square source = (attacker == kWhite ? destination + kSideSize : destination - kSideSize);
        moveList.push(Move(source, destination, kPawn, kKnight));
        moveList.push(Move(source, destination, kPawn, kBishop));
        moveList.push(Move(source, destination, kPawn, kRook));
        moveList.push(Move(source, destination, kPawn, kQueen));
      }

      // Push twice
      for (Bitboard sbb = (attacker == kWhite ? shiftUp(pushOnceBB) & kRank4Mask : shiftDown(pushOnceBB) & kRank5Mask) & ~bothOccupancy_;
           sbb;
           sbb = removeFirstPiece(sbb)) {
        Square destination = getFirstPieceSquare(sbb);
        Square source = (attacker == kWhite ? destination + 2 * kSideSize : destination - 2 * kSideSize);
        moveList.push(Move(source, destination, kPawn, 0, Move::kDoublePushFlag));
      }

      // Capture without promotion
      for (Bitboard sbb = bitboards_[attacker][kPawn] & (attacker == kWhite ? ~kRank7Mask : ~kRank2Mask);
           sbb;
           sbb = removeFirstPiece(sbb)) {
        Square source = getFirstPieceSquare(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[attacker][source] & occupancy_[defender]; dbb; dbb = removeFirstPiece(dbb)) {
          Square destination = getFirstPieceSquare(dbb);
          moveList.push(Move(source, destination, kPawn, 0, Move::kCaptureFlag));
        }
      }

      // Capture with promotion
      for (Bitboard sbb = bitboards_[attacker][kPawn] & (attacker == kWhite ? kRank7Mask : kRank2Mask);
           sbb;
           sbb = removeFirstPiece(sbb)) {
        Square source = getFirstPieceSquare(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[attacker][source] & occupancy_[defender]; dbb; dbb = removeFirstPiece(dbb)) {
          Square destination = getFirstPieceSquare(dbb);
          moveList.push(Move(source, destination, kPawn, kKnight, Move::kCaptureFlag));
          moveList.push(Move(source, destination, kPawn, kBishop, Move::kCaptureFlag));
          moveList.push(Move(source, destination, kPawn, kRook, Move::kCaptureFlag));
          moveList.push(Move(source, destination, kPawn, kQueen, Move::kCaptureFlag));
        }
      }

      // Enpassant
      if (enpassant_ != NO_SQUARE) {
        for (Bitboard enpassantBB = kAttackTable<kPawn>[defender][enpassant_] & bitboards_[attacker][kPawn];
             enpassantBB;
             enpassantBB = removeFirstPiece(enpassantBB)) {
          Square source = getFirstPieceSquare(enpassantBB);
          moveList.push(Move(source, enpassant_, kPawn, 0, Move::kCaptureFlag|Move::kEnpassantFlag));
        }
      }
    }

    // Piece moves
    getPseudoPieceMove<attacker, kKnight>(moveList);
    getPseudoPieceMove<attacker, kKing>(moveList);
    getPseudoPieceMove<attacker, kBishop>(moveList);
    getPseudoPieceMove<attacker, kRook>(moveList);
    getPseudoPieceMove<attacker, kQueen>(moveList);

    // Castle
    {
      constexpr Bitboard kingCastlePermission = (attacker == kWhite ? kWhiteKingCastlePermission : kBlackKingCastlePermission);
      constexpr Bitboard queenCastlePermission = (attacker == kWhite ? kWhiteQueenCastlePermission : kBlackQueenCastlePermission);
      constexpr Bitboard kingCastleOccupancy = (attacker == kWhite ? 0x6000000000000000ull : 0x60ul);
      constexpr Bitboard queenCastleOccupancy = (attacker == kWhite ? 0xE00000000000000ull : 0xEull);
      constexpr Square kingSourceSquare = (attacker == kWhite ? E1 : E8);
      constexpr Square kingSideMiddleSquare = (attacker == kWhite ? F1 : F8);
      constexpr Square kingSideDestinationSquare = (attacker == kWhite ? G1 : G8);
      constexpr Square queenSideMiddleSquare = (attacker == kWhite ? D1 : D8);
      constexpr Square queenSideDestinationSquare = (attacker == kWhite ? C1 : C8);

      // Check castle permission, blocker occupancy, and castle square safety.
      // Must check the king and the adjacent square to prevent illegal castle that makes king safe.
      // No need to check the king square after castle, because it will get verified by the legal move generator later.
      if (((castlePermission_ & kingCastlePermission) == kingCastlePermission) && (kingCastleOccupancy & bothOccupancy_) == 0 &&
          !isAttacked<defender>(kingSourceSquare) && !isAttacked<defender>(kingSideMiddleSquare)) {
        moveList.push(Move(kingSourceSquare, kingSideDestinationSquare, kKing, 0, Move::kCastlingFlag));
      }

      if (((castlePermission_ & queenCastlePermission) == queenCastlePermission) && (queenCastleOccupancy & bothOccupancy_) == 0 &&
          !isAttacked<defender>(kingSourceSquare) && !isAttacked<defender>(queenSideMiddleSquare)) {
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

  constexpr BoardState& currentState() {
    return currentState_;
  }

  constexpr const BoardState& currentState() const {
    return currentState_;
  }

  void setState(const BoardState& state) {
    currentState_ = state;
  }

  // Play the move and change the current state regardless its legality.
  // Return true if the move is legal, false otherwise.
  template <Team attacker>
  bool tryMakeMove(Move move) {
    constexpr Team defender = getOtherTeam(attacker);
    const Square source = move.getSourceSquare();
    const Square destination = move.getDestinationSquare();
    const Piece movedPiece = move.getMovedPiece();

    // Move the piece.
    auto& currentStateRef = currentState();
    currentStateRef.bitboards_[attacker][movedPiece] = moveSquare(currentStateRef.bitboards_[attacker][movedPiece], source, destination);
    currentStateRef.occupancy_[attacker] = moveSquare(currentStateRef.occupancy_[attacker], source, destination);
    currentStateRef.enpassant_ = NO_SQUARE;
    ++currentStateRef.halfmove_;
    if constexpr (attacker == kBlack) {
      ++currentStateRef.fullmove_;
    }

    if (move.isCaptured()) {
      currentStateRef.bitboards_[defender][kPawn] = unsetSquare(currentStateRef.bitboards_[defender][kPawn], destination);
      currentStateRef.bitboards_[defender][kKnight] = unsetSquare(currentStateRef.bitboards_[defender][kKnight], destination);
      currentStateRef.bitboards_[defender][kBishop] = unsetSquare(currentStateRef.bitboards_[defender][kBishop], destination);
      currentStateRef.bitboards_[defender][kRook] = unsetSquare(currentStateRef.bitboards_[defender][kRook], destination);
      currentStateRef.bitboards_[defender][kQueen] = unsetSquare(currentStateRef.bitboards_[defender][kQueen], destination);
      currentStateRef.occupancy_[defender] = unsetSquare(currentStateRef.occupancy_[defender], destination); // Does not affect enpassant
      currentStateRef.halfmove_ = 0;
    }

    if (movedPiece == kPawn) {
      currentStateRef.halfmove_ = 0;
    }

    currentStateRef.halfmove_ = 0;
    if (move.isDoublePush()) {
      currentStateRef.enpassant_ = (attacker == kWhite ? destination + 8 : destination - 8);

    } else if (move.isEnpassant()) {
      Square enpassantPawnSquare = (attacker == kWhite ? destination + 8 : destination - 8);
      currentStateRef.bitboards_[defender][kPawn] = unsetSquare(currentStateRef.bitboards_[defender][kPawn], enpassantPawnSquare);
      currentStateRef.occupancy_[defender] = unsetSquare(currentStateRef.occupancy_[defender], enpassantPawnSquare);

    } else if (Piece promotedPiece = move.getPromotedPiece(); promotedPiece) {
      currentStateRef.bitboards_[attacker][movedPiece] = unsetSquare(currentStateRef.bitboards_[attacker][movedPiece], destination);
      currentStateRef.bitboards_[attacker][promotedPiece] = setSquare(currentStateRef.bitboards_[attacker][promotedPiece], destination);

    } else if (move.isCastling()) {
      auto [rookSource, rookDestination] = [destination]() {
        switch (destination) {
        case G1: return std::pair<Square, Square>{ H1, F1 };
        case G8: return std::pair<Square, Square>{ H8, F8 };
        case C1: return std::pair<Square, Square>{ A1, D1 };
        case C8: return std::pair<Square, Square>{ A8, D8 };
        [[unlikely]] default:
          assert(false && "bad king castling square");
          return std::pair<Square, Square>{static_cast<Square>(0), static_cast<Square>(0)};
        }
      }();
      currentStateRef.bitboards_[attacker][kRook] = moveSquare(currentStateRef.bitboards_[attacker][kRook], rookSource, rookDestination);
      currentStateRef.occupancy_[attacker] = moveSquare(currentStateRef.occupancy_[attacker], rookSource, rookDestination);
    }

    // Update castle permission and occupancy
    currentStateRef.castlePermission_ = unsetSquare(unsetSquare(currentStateRef.castlePermission_, source), destination);
    currentStateRef.bothOccupancy_ = currentStateRef.occupancy_[kWhite] | currentStateRef.occupancy_[kBlack];

#ifdef _DEBUG
    // Check occupancy consistency.
    for (Team team : {kWhite, kBlack}) {
      Bitboard occupancy = 0;
      for (Bitboard bb : currentStateRef.bitboards_[team]) {
        occupancy |= bb;
      }
      if (occupancy != currentStateRef.occupancy_[team]) {
        std::cout << move.toString() << '\n';
        printBitboard(occupancy);
        printBitboard(currentStateRef.occupancy_[team]);
        assert(false && "occupancy inconsistent");
      }
    }
#endif

    currentStateRef.team_ = defender;

    // Check if the move is legal
    const Square kingSquare = getFirstPieceSquare(currentStateRef.bitboards_[attacker][kKing]);
    return !currentStateRef.isAttacked<defender>(kingSquare);
  }

public:
  explicit Board(const std::string& fen) {
    setFEN(fen);
  }

  void setFEN(const std::string& fen) {
    setState(BoardState::fromFEN(fen));
  }

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
