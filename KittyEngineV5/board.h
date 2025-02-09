#pragma once
#include "attack.h"
#include "move.h"
#include <array>
#include <format>
#include <iostream>

///////////////////////////////////////////////////////
//                 CHESS BOARD
///////////////////////////////////////////////////////

class Board {
  std::array<std::array<Bitboard, kPieceSize>, kTeamSize> bitboards_;
  std::array<Bitboard, kTeamSize> occupancy_;
  Bitboard bothOccupancy_;
  Team team_;
  CastlePermission castlePermission_;
  uint32_t enpassant_;
  uint32_t halfmove_;
  uint32_t fullmove_;

  template <Team attacker>
  constexpr bool isSquareAttacked(uint32_t square) const {
    // Simulate all attacks at the square.
    // If you can attack the enemy piece with the same attack type, then they can attack you.
    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);
    const Bitboard bishopAttack = SliderTable::getAttack<kBishop>(square, bothOccupancy_);
    const Bitboard rookAttack = SliderTable::getAttack<kRook>(square, bothOccupancy_);
    const Bitboard queenAttack = bishopAttack | rookAttack;

    return  kAttackTable<kPawn>[defender][square] & bitboards_[attacker][kPawn] |
      kAttackTable<kKnight>[square] & bitboards_[attacker][kKnight] |
      kAttackTable<kKing>[square] & bitboards_[attacker][kKing] |
      bishopAttack & bitboards_[attacker][kBishop] |
      rookAttack & bitboards_[attacker][kRook] |
      queenAttack & bitboards_[attacker][kQueen];
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

public:
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
  static Board fromFEN(const std::string& fen);
  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
