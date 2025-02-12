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
  Team team_;
  uint32_t castlePermission_;
  uint32_t enpassant_;
  uint16_t halfmove_;
  uint16_t fullmove_;

  template <Team attacker>
  constexpr bool isSquareAttacked(uint32_t square) const {
    // Simulate all attacks at the square.
    // If you can attack the enemy piece with the same attack type, then they can attack you.
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);
    if ((bitboards_[attacker][kPawn] & kAttackTable<kPawn>[defender][square]) |
        (bitboards_[attacker][kKnight] & kAttackTable<kKnight>[square]) |
        (bitboards_[attacker][kKing] & kAttackTable<kKing>[square])) {
      return true;
    }

    const Bitboard bishopAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    if (bitboards_[attacker][kBishop] & bishopAttack) {
      return true;
    }

    const Bitboard rookAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    if (bitboards_[attacker][kRook] & rookAttack) {
      return true;
    }

    const Bitboard queenAttack = bishopAttack | rookAttack;
    return bitboards_[attacker][kQueen] & queenAttack;
  }

  template <Team attacker, Piece piece>
  constexpr void getPseudoPieceMove(MoveList& moveList) const {
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);

    for (Bitboard sbb = bitboards_[attacker][piece];
         sbb;
         sbb = removeFirstPiece(sbb)) {
      uint32_t source = findFirstPiece(sbb);
      Bitboard attackBB;
      if constexpr (piece == kBishop || piece == kRook || piece == kQueen) {
        attackBB = SliderTable::getAttack<piece>(source, bothOccupancy_);
      } else {
        attackBB = kAttackTable<piece>[source];
      }

      // Non-capture
      for (Bitboard dbb = attackBB & ~bothOccupancy_;
           dbb;
           dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        moveList.push(Move(source, destination, piece));
      }

      // Capture
      for (Bitboard dbb = attackBB & occupancy_[defender];
           dbb;
           dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        moveList.push(Move(source, destination, piece, 0, Move::kCaptureFlag));
      }
    }
  }

  // TODO: consider put NO_SQUARE in kPawnAttackTable and give it an empty mask
  // TODO: separate capture move, promotion, and quiet move
  // TODO: generate all pawn moves at once
  template <Team attacker>
  constexpr void getPseudoMove(MoveList& moveList) const {
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);

    // Pawn
    {
      const Bitboard pushOnceBB = (attacker == kWhite ? shiftUp(bitboards_[attacker][kPawn]) : shiftDown(bitboards_[attacker][kPawn])) & ~bothOccupancy_;

      // Push once without promotion
      for (Bitboard dbb = pushOnceBB & (attacker == kWhite ? ~kRank8Mask : ~kRank1Mask);
           dbb;
           dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        uint32_t source = (attacker == kWhite ? destination + kSideSize : destination - kSideSize);
        moveList.push(Move(source, destination, kPawn));
      }

      // Push once with promotion
      for (Bitboard dbb = pushOnceBB & (attacker == kWhite ? kRank8Mask : kRank1Mask);
           dbb;
           dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        uint32_t source = (attacker == kWhite ? destination + kSideSize : destination - kSideSize);
        moveList.push(Move(source, destination, kPawn, kKnight));
        moveList.push(Move(source, destination, kPawn, kBishop));
        moveList.push(Move(source, destination, kPawn, kRook));
        moveList.push(Move(source, destination, kPawn, kQueen));
      }

      // Push twice
      for (Bitboard sbb = (attacker == kWhite ? shiftUp(pushOnceBB) & kRank4Mask : shiftDown(pushOnceBB) & kRank5Mask) & ~bothOccupancy_;
           sbb;
           sbb = removeFirstPiece(sbb)) {
        uint32_t destination = findFirstPiece(sbb);
        uint32_t source = (attacker == kWhite ? destination + 2 * kSideSize : destination - 2 * kSideSize);
        moveList.push(Move(source, destination, kPawn, 0, Move::kDoublePushFlag));
      }

      // Capture without promotion
      for (Bitboard sbb = bitboards_[attacker][kPawn] & (attacker == kWhite ? ~kRank7Mask : ~kRank2Mask);
           sbb;
           sbb = removeFirstPiece(sbb)) {
        uint32_t source = findFirstPiece(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[attacker][source] & occupancy_[defender]; dbb; dbb = removeFirstPiece(dbb)) {
          uint32_t destination = findFirstPiece(dbb);
          moveList.push(Move(source, destination, kPawn, 0, Move::kCaptureFlag));
        }
      }

      // Capture with promotion
      for (Bitboard sbb = bitboards_[attacker][kPawn] & (attacker == kWhite ? kRank7Mask : kRank2Mask);
           sbb;
           sbb = removeFirstPiece(sbb)) {
        uint32_t source = findFirstPiece(sbb);
        for (Bitboard dbb = kAttackTable<kPawn>[attacker][source] & occupancy_[defender]; dbb; dbb = removeFirstPiece(dbb)) {
          uint32_t destination = findFirstPiece(dbb);
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
          uint32_t source = findFirstPiece(enpassantBB);
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
      constexpr uint32_t kingCastlePermission = (attacker == kWhite ? kWhiteKingCastlePermission : kBlackKingCastlePermission);
      constexpr uint32_t queenCastlePermission = (attacker == kWhite ? kWhiteQueenCastlePermission : kBlackQueenCastlePermission);
      constexpr Bitboard kingCastleOccupancy = (attacker == kWhite ? 0x6000000000000000ull : 0x60ul);
      constexpr Bitboard queenCastleOccupancy = (attacker == kWhite ? 0xE00000000000000ull : 0xEull);
      constexpr uint32_t kingSourceSquare = (attacker == kWhite ? E1 : E8);
      constexpr uint32_t kingSideMiddleSquare = (attacker == kWhite ? F1 : F8);
      constexpr uint32_t kingSideDestinationSquare = (attacker == kWhite ? G1 : G8);
      constexpr uint32_t queenSideMiddleSquare = (attacker == kWhite ? D1 : D8);
      constexpr uint32_t queenSideDestinationSquare = (attacker == kWhite ? C1 : C8);

      // Check castle permission, blocker occupancy, and castle square safety.
      // Must check the king and the adjacent square to prevent illegal castle that makes king safe.
      // No need to check the king square after castle, because it will get verified by the legal move generator later.
      if ((castlePermission_ & kingCastlePermission) && (kingCastleOccupancy & bothOccupancy_) == 0 &&
          !isSquareAttacked<defender>(kingSourceSquare) && !isSquareAttacked<defender>(kingSideMiddleSquare)) {
        moveList.push(Move(kingSourceSquare, kingSideDestinationSquare, kKing, 0, Move::kCastlingFlag));
      }

      if ((castlePermission_ & queenCastlePermission) && (queenCastleOccupancy & bothOccupancy_) == 0 &&
          !isSquareAttacked<defender>(kingSourceSquare) && !isSquareAttacked<defender>(queenSideMiddleSquare)) {
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
  std::array<BoardState, 512> states_;
  size_t stateSize_;

  constexpr BoardState& currentState() {
    return states_[stateSize_ - 1];
  }

  constexpr const BoardState& currentState() const {
    return states_[stateSize_ - 1];
  }

  void pushState() {
    states_[stateSize_] = states_[stateSize_ - 1];
    ++stateSize_;
  }

  void popState() {
    --stateSize_;
  }

  template <Team attacker>
  bool tryMakeMove(Move move) {
    pushState();

    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);
    const uint32_t source = move.getSourceSquare();
    const uint32_t destination = move.getDestinationSquare();
    const uint32_t movedPiece = move.getMovedPiece();

    // Move the piece.
    auto& currentStateRef = currentState();
    currentStateRef.bitboards_[attacker][movedPiece] = moveSquare(currentStateRef.bitboards_[attacker][movedPiece], source, destination);
    currentStateRef.occupancy_[attacker] = moveSquare(currentStateRef.occupancy_[attacker], source, destination);

    if (move.isCaptured()) {
      currentStateRef.bitboards_[defender][kPawn] = unsetSquare(currentStateRef.bitboards_[defender][kPawn], destination);
      currentStateRef.bitboards_[defender][kKnight] = unsetSquare(currentStateRef.bitboards_[defender][kKnight], destination);
      currentStateRef.bitboards_[defender][kBishop] = unsetSquare(currentStateRef.bitboards_[defender][kBishop], destination);
      currentStateRef.bitboards_[defender][kRook] = unsetSquare(currentStateRef.bitboards_[defender][kRook], destination);
      currentStateRef.bitboards_[defender][kQueen] = unsetSquare(currentStateRef.bitboards_[defender][kQueen], destination);
      currentStateRef.occupancy_[defender] = unsetSquare(currentStateRef.occupancy_[defender], destination); // Does not affect enpassant
    }

    if (uint32_t promotedPiece = move.getPromotedPiece(); promotedPiece) {
      currentStateRef.bitboards_[attacker][movedPiece] = unsetSquare(currentStateRef.bitboards_[attacker][movedPiece], destination);
      currentStateRef.bitboards_[attacker][promotedPiece] = setSquare(currentStateRef.bitboards_[attacker][promotedPiece], destination);
    }

    if (move.isEnpassant()) {
      uint32_t enpassantPawnSquare = (attacker == kWhite ? destination + 8 : destination - 8);
      currentStateRef.bitboards_[defender][kPawn] = unsetSquare(currentStateRef.bitboards_[defender][kPawn], enpassantPawnSquare);
      currentStateRef.occupancy_[defender] = unsetSquare(currentStateRef.occupancy_[defender], enpassantPawnSquare);
    }

    if (move.isDoublePush()) {
      currentStateRef.enpassant_ = (attacker == kWhite ? destination + 8 : destination - 8);
    } else {
      currentStateRef.enpassant_ = NO_SQUARE;
    }

    if (move.isCastling()) {
      auto[rookSource, rookDestination] = [destination]() {
        switch (destination) {
        case G1: return std::pair<uint32_t, uint32_t>{ H1, F1 };
        case G8: return std::pair<uint32_t, uint32_t>{ H8, F8 };
        case C1: return std::pair<uint32_t, uint32_t>{ A1, D1 };
        case C8: return std::pair<uint32_t, uint32_t>{ A8, D8 };
        [[unlikely]] default: 
          assert(false && "bad king castling square");
          return std::pair<uint32_t, uint32_t>{0, 0};
        }
      }();
      currentStateRef.bitboards_[attacker][kRook] = moveSquare(currentStateRef.bitboards_[attacker][kRook], rookSource, rookDestination);
      currentStateRef.occupancy_[attacker] = moveSquare(currentStateRef.occupancy_[attacker], rookSource, rookDestination);
    }

    // Update castle permission
    currentStateRef.castlePermission_ &= kCastlePermissionUpdateTable[source] & kCastlePermissionUpdateTable[destination];

    // Update the occupancy
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

    // Update 50 moves rules
    if (movedPiece == kPawn || move.isCaptured()) {
      currentStateRef.halfmove_ = 0;
    } else {
      ++currentStateRef.halfmove_;
    }
    if constexpr (attacker == kBlack) {
      ++currentStateRef.fullmove_;
    }

    // Check if the move is legal
    const uint32_t kingSquare = findFirstPiece(currentStateRef.bitboards_[attacker][kKing]);
    if (currentStateRef.isSquareAttacked<defender>(kingSquare)) {
      popState();
      return false;
    }

    currentStateRef.team_ = defender;
    return true;
  }

public:
  explicit Board(const std::string& fen) {
    setFEN(fen);
  }

  void setFEN(const std::string& fen) {
    stateSize_ = 1;
    currentState() = BoardState::fromFEN(fen);
  }

  // https://www.chessprogramming.org/Perft_Results
  static inline const std::string kStartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  static inline const std::string kKiwipeteFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
  static inline const std::string kRookEndGameFEN = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - ";
  static inline const std::string kMirrorViewFEN = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
  static inline const std::string kTalkChessBugFEN = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
  static inline const std::string kStevenAltFEN = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
