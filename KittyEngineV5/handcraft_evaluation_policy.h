#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include "evaluation_policy.h"
#include <algorithm>
#include <array>
#include <cstdint>

namespace bb::evaluation {
  class HandCraftEvaluationPolicy {
    static constexpr int32_t kDoubledPawnPenalty = -31;
    static constexpr int32_t kIsolatedPawnPenalty = -11;
    static constexpr int32_t kMissingShelterPenalty = -8;
    static constexpr int32_t kSemiOpenKingFilePenalty = 0;
    static constexpr int32_t kOpenKingFilePenalty = -10;
    static constexpr int32_t kKingZoneAttackPenalty = -4;
    static constexpr int32_t kKingZoneDoubleAttackPenalty = -12;
    static constexpr int32_t kBishopPairBonus = 37;
    static constexpr int32_t kSemiOpenFileBonus = 9;
    static constexpr int32_t kOpenFileBonus = 24;
    static constexpr int32_t kMinKingWeaknessPenalty = -92;
    static constexpr int32_t kMaxPhase = 24;

    static constexpr std::array<int32_t, kBoardLenSize> kPassedPawnBonusTable = { 0, 0, 5, 8, 31, 46, 118, 0 };
    static constexpr std::array<int32_t, kPieceSize> kPieceMobilityBonusTable = { 0, 4, 6, 2, 0, 0 };
    static constexpr std::array<int32_t, kPieceSize> kPieceMaterialValueTable = { 100, 309, 291, 490, 951, 0 };

    // Piece-square tables for 6 piece types from White's perspective.
    static constexpr std::array<std::array<int32_t, kSquareSize>, kPieceSize> kPiecePositionalValueTable = { {
        {}, // Pawn (uses game-stage tables below)
        { // Knight
            -40,-30,-25,-25,-25,-25,-30,-40,
            -30,-15,  0,  0,  0,  0,-15,-30,
            -25,  0, 10, 12, 12, 10,  0,-25,
            -25,  5, 12, 15, 15, 12,  5,-25,
            -25,  0, 12, 15, 15, 12,  0,-25,
            -25,  5, 10, 12, 12, 10,  5,-25,
            -30,-15,  0,  5,  5,  0,-15,-30,
            -40,-30,-25,-25,-25,-25,-30,-40
        },
        { // Bishop
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10, 15,  0,  0,  0,  0, 15,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        },
        { // Rook
             15, 15, 15, 15, 15, 15, 15, 15,
             15, 15, 15, 15, 15, 15, 15, 15,
              0,  5,  7, 10, 10,  7,  5,  0,
              0,  5,  7, 10, 10,  7,  5,  0,
              0,  5,  7, 10, 10,  7,  5,  0,
              0,  5,  7, 10, 10,  7,  5,  0,
              0,  5,  7, 10, 10,  7,  5,  0,
              0,  0,  0,  8,  8,  0,  0,  0
        },
        { // Queen
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
             -5,  0,  0,  0,  0,  0,  0, -5,
             -5,  0,  0,  0,  0,  0,  0, -5,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        },
        {} // King (uses game-stage tables below)
    }};

    // [Stage 0: Early Game, 1: Late Game][Square 0..63]
    static constexpr int32_t kPawnPositionalValueTable[2][64] = {
        { // Early Game
              0,  0,  0,  0,  0,  0,  0,  0,
             22, 24, 26, 28, 28, 26, 24, 22,
             19, 21, 23, 25, 25, 23, 21, 19,
             12, 12, 15, 22, 22, 15, 12, 12,
              5,  5,  0, 20, 20,  5,  5,  5,
              5,  0, -8,  0,  0,-10,  0,  5,
              5, 10,  0,-20,-20,  0, 10,  5,
              0,  0,  0,  0,  0,  0,  0,  0
        },
        { // Late Game
              0,  0,  0,  0,  0,  0,  0,  0,
             24, 26, 28, 30, 30, 28, 26, 24,
             19, 21, 23, 25, 25, 23, 21, 19,
             14, 16, 18, 20, 20, 18, 16, 14,
              9, 11, 13, 15, 15, 13, 11,  9,
              4,  6,  8, 10, 10,  8,  6,  4,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0
        }
    };

    // [Stage 0: Early Game, 1: End Game][Square 0..63]
    static constexpr int32_t kKingPositionalValueTable[2][64] = {
        { // Early Game
            -25,-25,-25,-25,-25,-25,-25,-25,
            -25,-25,-25,-25,-25,-25,-25,-25,
            -25,-25,-25,-25,-25,-25,-25,-25,
            -20,-20,-20,-25,-25,-20,-20,-20,
            -15,-15,-15,-25,-25,-15,-15,-15,
            -10,-10,-10,-20,-20,-10,-10,-10,
            -10,-10,  0,-15,-15,  0,-10,-10,
             20, 30,  0,  0,  0,  0, 30, 20
        },
        { // End Game
            -35,-20,-20,-20,-20,-20,-20,-35,
            -20,-15,-10,-10,-10,-10,-15,-20,
            -20,-10, 10, 10, 10, 10,-10,-20,
            -20,-10, 10, 12, 12, 10,-10,-20,
            -20,-10, 10, 12, 12, 10,-10,-20,
            -20,-10, 10, 10, 10, 10,-10,-20,
            -20,-15,-10,-10,-10,-10,-15,-20,
            -35,-20,-20,-20,-20,-20,-20,-35
        }
    };

    // Precomputed adjacent file bitboards for each of the 8 board files (0..7)
    static constexpr std::array<Bitboard, 8> kAdjacentFileMasks = {
      kFileBMask,
      kFileAMask | kFileCMask,
      kFileBMask | kFileDMask,
      kFileCMask | kFileEMask,
      kFileDMask | kFileFMask,
      kFileEMask | kFileGMask,
      kFileFMask | kFileHMask,
      kFileGMask
    };

    // Precomputed passed pawn masks (enemy opposing squares ahead on same/adjacent files)
    static constexpr auto kPassedPawnMasks = [] {
      std::array<std::array<Bitboard, kSquareSize>, kSideSize> masks{};
      for (Square sq = 0; sq < kSquareSize; ++sq) {
        const int r = static_cast<int>(getSquareRank(sq));
        const int f = static_cast<int>(getSquareFile(sq));
        for (int tr = 0; tr < 8; ++tr) {
          for (int tf = std::max(0, f - 1); tf <= std::min(7, f + 1); ++tf) {
            const Bitboard target = 1ull << (tr * 8 + tf);
            if (tr < r) {
              masks[White][sq] |= target;
            }
            if (tr > r) {
              masks[Black][sq] |= target;
            }
          }
        }
      }
      return masks;
    }();

    template <Side side>
    static constexpr Square orient(Square sq) noexcept {
      if constexpr (side == White) return sq; else return sq ^ 56;
    }

    static constexpr int32_t blend(int32_t early, int32_t late, int32_t phase) noexcept {
      return (early * phase + late * (kMaxPhase - phase)) / kMaxPhase;
    }

    struct AttackInfo {
      Bitboard once = 0;
      Bitboard twice = 0;

      void add(Bitboard mask) noexcept {
        twice |= once & mask;
        once |= mask;
      }
    };

    template <Side side>
    static int32_t scoreNonPawns(const BoardState& state, Bitboard bothOccupancy,
                                 AttackInfo& attacks) noexcept {
      const Bitboard notOwn = ~state.getOccupancy(side);
      const Bitboard ownPawns = state.getPieces(side, Pawn);
      const Bitboard enemyPawns = state.getPieces(getOtherSide(side), Pawn);
      int32_t score = 0;

      // Evaluate each piece type: material, PST, mobility, and rook file openness.
      auto evaluatePiece = [&]<Piece piece>() {
        for (Bitboard bb = state.getPieces(side, piece); bb; bb = popPiece(bb)) {
          const Square sq = peekPiece(bb);
          score += kPieceMaterialValueTable[piece] +
            kPiecePositionalValueTable[piece][orient<side>(sq)];

          // Generate attacks (knight lookup or magic slider rays) and accumulate coverage.
          Bitboard attackMask;
          if constexpr (piece == Knight) {
            attackMask = internal::kKnightAttackTable[sq];
          } else {
            attackMask = getAttack<piece>(sq, bothOccupancy);
          }
          attacks.add(attackMask);

          // Mobility bonus for non-queen pieces on unblocked squares.
          if constexpr (piece != Queen) {
            score += static_cast<int32_t>(countPiece(attackMask & notOwn)) *
              kPieceMobilityBonusTable[piece];
          }

          // Bonus for rooks on open or semi-open files.
          if constexpr (piece == Rook) {
            const Bitboard file = kSquareToFileMasks[sq];
            if ((ownPawns & file) == 0) {
              score += (enemyPawns & file) == 0 ? kOpenFileBonus : kSemiOpenFileBonus;
            }
          }
        }
      };

      evaluatePiece.template operator()<Knight>();
      evaluatePiece.template operator()<Bishop>();
      evaluatePiece.template operator()<Rook>();
      evaluatePiece.template operator()<Queen>();
      return score;
    }

    template <Side side>
    static int32_t scoreSide(const BoardState& state, Bitboard bothOccupancy,
                             int32_t phase, AttackInfo& attacks) noexcept {
      const Bitboard pawns = state.getPieces(side, Pawn);
      const Bitboard enemyPawns = state.getPieces(getOtherSide(side), Pawn);
      int32_t score = 0;

      // Score each pawn: material, tapered PST, isolated pawns, and passed pawns.
      for (Bitboard p = pawns; p; p = popPiece(p)) {
        const Square sq = peekPiece(p);
        const Square oriented = orient<side>(sq);
        score += kPieceMaterialValueTable[Pawn] +
          blend(kPawnPositionalValueTable[0][oriented],
                kPawnPositionalValueTable[1][oriented], phase);

        // Isolated pawn: no friendly pawns on adjacent files.
        if ((pawns & kAdjacentFileMasks[getSquareFile(sq)]) == 0) {
          score += kIsolatedPawnPenalty;
        }

        // Passed pawn: no opposing enemy pawns in the files ahead.
        if ((enemyPawns & kPassedPawnMasks[side][sq]) == 0) {
          score += kPassedPawnBonusTable[7 - getSquareRank(oriented)];
        }
      }

      // Doubled pawns: vertical bitboard smear counts extra pawns sharing files.
      Bitboard smeared = pawns;
      smeared |= smeared >> 8;
      smeared |= smeared >> 16;
      smeared |= smeared >> 32;
      const int32_t occupiedFiles = static_cast<int32_t>(countPiece(smeared & kRank8Mask));
      score += (static_cast<int32_t>(countPiece(pawns)) - occupiedFiles) * kDoubledPawnPenalty;

      // Pawn attacks: diagonal bit shifts update attack coverage in parallel.
      if constexpr (side == White) {
        attacks.add(shiftUpLeft(pawns));
        attacks.add(shiftUpRight(pawns));
      } else {
        attacks.add(shiftDownLeft(pawns));
        attacks.add(shiftDownRight(pawns));
      }

      // Non-pawn pieces
      score += scoreNonPawns<side>(state, bothOccupancy, attacks);

      // Bishop pair
      if (countPiece(state.getPieces(side, Bishop)) >= 2) {
        score += kBishopPairBonus;
      }

      // King placement
      const Square kingSquare = peekPiece(state.getPieces(side, King));
      const Square orientedKing = orient<side>(kingSquare);
      score += blend(kKingPositionalValueTable[0][orientedKing],
                     kKingPositionalValueTable[1][orientedKing], phase);
      attacks.add(internal::kKingAttackTable[kingSquare]);

      return score;
    }

    template <Side side>
    static int32_t kingWeakness(const BoardState& state, const AttackInfo& enemyAttacks,
                                int32_t phase) noexcept {
      if (phase == 0) {
        return 0;
      }
      const Square kingSquare = peekPiece(state.getPieces(side, King));
      const int kingRank = static_cast<int>(getSquareRank(kingSquare));
      const int kingFile = static_cast<int>(getSquareFile(kingSquare));
      int ahead1;
      int ahead2;
      if constexpr (side == White) {
        ahead1 = kingRank - 1;
        ahead2 = kingRank - 2;
      } else {
        ahead1 = kingRank + 1;
        ahead2 = kingRank + 2;
      }
      const Bitboard ownPawns = state.getPieces(side, Pawn);
      const Bitboard enemyPawns = state.getPieces(getOtherSide(side), Pawn);
      int32_t penalty = 0;

      // King shelter: check pawn shield in front of king and open/semi-open files.
      for (int file = std::max(0, kingFile - 1); file <= std::min(7, kingFile + 1); ++file) {
        Bitboard shelter = 0;
        if (ahead1 >= 0 && ahead1 < 8) shelter |= (1ull << (ahead1 * 8 + file));
        if (ahead2 >= 0 && ahead2 < 8) shelter |= (1ull << (ahead2 * 8 + file));

        if ((ownPawns & shelter) == 0) {
          penalty += kMissingShelterPenalty;
        }

        const Bitboard fileMask = kFileAMask << file;
        if ((ownPawns & fileMask) == 0) {
          penalty += (enemyPawns & fileMask) == 0 ? kOpenKingFilePenalty : kSemiOpenKingFilePenalty;
        }
      }

      // King safety: penalize enemy attack rays landing in the king's immediate zone.
      const Bitboard kingZone = internal::kKingAttackTable[kingSquare] | (1ull << kingSquare);
      penalty += kKingZoneAttackPenalty * static_cast<int32_t>(countPiece(enemyAttacks.once & kingZone));
      penalty += kKingZoneDoubleAttackPenalty * static_cast<int32_t>(countPiece(enemyAttacks.twice & kingZone));
      return std::max(penalty, kMinKingWeaknessPenalty) * phase / kMaxPhase;
    }

  public:
    void reset() noexcept {}
    template <bool Add> void markPiece(Side, Piece, Square) noexcept {}

    int32_t evaluate(const BoardState& state) const noexcept {
      // Calculate game phase (0 = endgame, 24 = opening/middlegame).
      const int32_t phase = std::min(kMaxPhase,
        static_cast<int32_t>(
          countPiece(state.getPieces(White, Knight) | state.getPieces(Black, Knight)) +
          countPiece(state.getPieces(White, Bishop) | state.getPieces(Black, Bishop)) +
          2 * countPiece(state.getPieces(White, Rook) | state.getPieces(Black, Rook)) +
          4 * countPiece(state.getPieces(White, Queen) | state.getPieces(Black, Queen))));

      // Dead position: bare kings or lone minor without pawns
      if (phase <= 1 && (state.getPieces(White, Pawn) | state.getPieces(Black, Pawn)) == 0) {
        return 0;
      }

      const Bitboard bothOccupancy = state.getOccupancy(White) | state.getOccupancy(Black);
      AttackInfo whiteAttacks;
      AttackInfo blackAttacks;

      // Net score: White evaluation minus Black evaluation, plus king safety terms.
      const int32_t netScore = scoreSide<White>(state, bothOccupancy, phase, whiteAttacks) -
                               scoreSide<Black>(state, bothOccupancy, phase, blackAttacks) +
                               kingWeakness<White>(state, blackAttacks, phase) -
                               kingWeakness<Black>(state, whiteAttacks, phase);

      return state.getSide() == White ? netScore : -netScore;
    }
  };

  static_assert(EvaluationPolicy<HandCraftEvaluationPolicy>, "Not satisfy EvaluationPolicy concept");
}
