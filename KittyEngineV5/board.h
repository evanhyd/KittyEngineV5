#pragma once
#include "bitboard.h"
#include <array>
#include <iosfwd>

struct Move {
  uint32_t source;
  uint32_t destination;
};

class MoveStack {
  static constexpr size_t kMaxMovePerTurn = 218;
  std::array<Move, kMaxMovePerTurn> moves{};
  uint32_t size = 0;

public:
  void Append(const Move& move) {
    moves[size] = move;
    ++size;
  }
};

class Board {

  std::array<std::array<Bitboard, kPieceSize>, kTeamSize> bitboards_;
  std::array<Bitboard, kTeamSize> occupancy_;
  Team team_;
  CastlePermission castlePermission_;
  Square enpassant_;
  uint32_t halfmove_;
  uint32_t fullmove_;

  template <Team attacker>
  bool isSquareAttacked(uint32_t square) const {
    // If your piece can attack square x,
    // then the enemy piece with same type on square x can attack back.
    constexpr Team defender = !attacker;

    Bitboard bishopAttack = Magic::getBishopAttack(square, occupancy_[defender]);
    Bitboard rookAttack = Magic::getRookAttack(square, occupancy_[defender]);
    Bitboard queenAttack = bishopAttack | rookAttack;

    return (kPawnAttackTable[defender][square] & bitboards_[attacker][kPawn] |
            kKnightAttackTable[square] & bitboards_[attacker][kKnight] |
            kKingAttackTable[square] & bitboards_[attacker][kKing] |
            bishopAttack & bitboards_[attacker][kBishop] |
            rookAttack & bitboards_[attacker][kRook] | 
            queenAttack & bitboards_[attacker][kQueen]) >> square & 1;
  }

  template <Team teamToPlay>
  void generatePseudoMove(MoveStack& moves) {
    const Bitboard notOccupiedByAlly = ~occupancy_[teamToPlay];

    // Knight
    for (Bitboard sbb = bitboards_[teamToPlay][kKnight]; sbb; sbb = removeFirstPiece(sbb)) {
      uint32_t source = findFirstPiece(sbb);
      for (Bitboard dbb = kKnightAttackTable[source] & notOccupiedByAlly; dbb; dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        moves.Append(Move{ source, destination });
      }
    }

    // King
    for (Bitboard sbb = bitboards_[teamToPlay][kKing]; sbb; sbb = removeFirstPiece(sbb)) {
      uint32_t source = findFirstPiece(sbb);
      for (Bitboard dbb = kKnightAttackTable[source] & notOccupiedByAlly; dbb; dbb = removeFirstPiece(dbb)) {
        uint32_t destination = findFirstPiece(dbb);
        moves.Append(Move{ source, destination });
      }
    }
  }

public:
  static Board fromFEN(const std::string& fen);
  friend std::ostream& operator<<(std::ostream& out, const Board& board);
};
