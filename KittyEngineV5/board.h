#pragma once
#include "attack.h"
#include "move.h"
#include <array>
#include <type_traits>

///////////////////////////////////////////////////////
//                 CHESS BOARD STATE
///////////////////////////////////////////////////////
class BoardState {
  std::array<std::array<Bitboard, kPieceSize>, kTeamSize> bitboards_;
  std::array<Bitboard, kTeamSize> occupancy_;
  Bitboard bothOccupancy_;
  Team team_;
  CastlePermission castlePermission_;
  uint32_t enpassant_;
  uint16_t halfmove_;
  uint16_t fullmove_;

  template <Team attacker>
  constexpr bool isSquareAttacked(uint32_t square) const {
    // Simulate all attacks at the square.
    // If you can attack the enemy piece with the same attack type, then they can attack you.
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);
    const Bitboard bishopAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    const Bitboard rookAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    const Bitboard queenAttack = bishopAttack | rookAttack;

    return  bitboards_[attacker][kPawn] & kAttackTable<kPawn>[defender][square] |
            bitboards_[attacker][kKnight] & kAttackTable<kKnight>[square] |
            bitboards_[attacker][kKing] & kAttackTable<kKing>[square] |
            bitboards_[attacker][kBishop] & bishopAttack |
            bitboards_[attacker][kRook] & rookAttack |
            bitboards_[attacker][kQueen] & queenAttack;
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
      constexpr CastlePermission kingCastlePermission = (attacker == kWhite ? kWhiteKingCastlePermission : kBlackKingCastlePermission);
      constexpr CastlePermission queenCastlePermission = (attacker == kWhite ? kWhiteQueenCastlePermission : kBlackQueenCastlePermission);
      constexpr Bitboard kingCastleOccupancy = (attacker == kWhite ? 6917529027641081856ull : 96ul);
      constexpr Bitboard queenCastleOccupancy = (attacker == kWhite ? 1008806316530991104ull : 14ul);
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
  }

  static inline const std::string kStartFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  static inline const std::string kKiwipeteFEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - ";
  static inline const std::string kEnpassantFEN = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
  static inline const std::string kWhiteKingCastleFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1";
  static inline const std::string kWhiteQueenCastleFEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1";
  static inline const std::string kNoCastleFEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1";
  static inline const std::string kCastleBlockedByCheckFEN = "rnb1kbnr/pppp1ppp/8/4p3/2B1P3/3P3q/PPP2P1P/RNBQK2R w KQkq - 0 5";
  static BoardState fromFEN(const std::string& fen);

  friend class Board;
  friend std::ostream& operator<<(std::ostream& out, const BoardState& boardState);
};

static_assert(std::is_trivial_v<BoardState>, "BoardState is not POD type, may affect performance");

class Board {
  BoardState currentState_;
  std::array<BoardState, 512> states_;
  size_t stateSize_;

  void pushState() {
    states_[stateSize_] = currentState_;
    ++stateSize_;
  }

  void popState() {
    --stateSize_;
    currentState_ = states_[stateSize_];
  }

  void setFEN(const std::string& fen) {
    currentState_ = BoardState::fromFEN(fen);
    stateSize_ = 0;
  }

  template <Team attacker>
  bool tryMakeMove(Move move) {
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);

    pushState();

    const uint32_t source = move.getSourceSquare();
    const uint32_t destination = move.getDestinationSquare();
    const uint32_t movedPiece = move.getMovedPiece();

    // Move the piece.
    currentState_.bitboards_[attacker][movedPiece] = moveSquare(currentState_.bitboards_[attacker][movedPiece], source, destination);
    currentState_.occupancy_[attacker] = moveSquare(currentState_.occupancy_[attacker], source, destination);

    if (move.isCaptured()) {
      // TODO: cache ~(1 << destination)
      currentState_.bitboards_[defender][kPawn] = unsetSquare(currentState_.bitboards_[defender][kPawn], destination);
      currentState_.bitboards_[defender][kKnight] = unsetSquare(currentState_.bitboards_[defender][kKnight], destination);
      currentState_.bitboards_[defender][kBishop] = unsetSquare(currentState_.bitboards_[defender][kBishop], destination);
      currentState_.bitboards_[defender][kRook] = unsetSquare(currentState_.bitboards_[defender][kRook], destination);
      currentState_.bitboards_[defender][kQueen] = unsetSquare(currentState_.bitboards_[defender][kQueen], destination);
      currentState_.occupancy_[defender] = unsetSquare(currentState_.occupancy_[defender], destination);
    }

    if (uint32_t promotedPiece = move.getPromotedPiece(); promotedPiece) {
      currentState_.bitboards_[attacker][movedPiece] = unsetSquare(currentState_.bitboards_[attacker][movedPiece], source);
      currentState_.bitboards_[attacker][promotedPiece] = setSquare(currentState_.bitboards_[attacker][promotedPiece], source);
    }

    if (move.isEnpassant()) {
      uint32_t enpassantPawnSquare = (attacker == kWhite ? destination + 8 : destination - 8);
      currentState_.bitboards_[defender][kPawn] = unsetSquare(currentState_.bitboards_[defender][kPawn], enpassantPawnSquare);
      currentState_.occupancy_[defender] = unsetSquare(currentState_.occupancy_[defender], enpassantPawnSquare);
    }

    if (move.isDoublePush()) {
      currentState_.enpassant_ = (attacker == kWhite ? destination + 8 : destination - 8);
    } else {
      currentState_.enpassant_ = NO_SQUARE;
    }

    if (move.isCastling()) {
      std::pair<uint32_t, uint32_t> rookMove = [destination]() {
        switch (destination) {
        case G1: return { H1, F1 };
        case G8: return { H8, F8 };
        case C1: return { A1, D1 };
        case C8: return { A8, D8 };
        }
      }();

      currentState_.bitboards_[attacker][kRook] = moveSquare(currentState_.bitboards_[attacker][kRook], rookMove.first, rookMove.second);
      currentState_.occupancy_[attacker] = move(currentState_.occupancy_[attacker], rookMove.first, rookMove.second);
    }

    currentState_.bothOccupancy_ = currentState_.occupancy_[kWhite] | currentState_.occupancy_[kBlack];
  }
};
