#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include <algorithm>
#include <array>
#include <cstdint>

namespace bb::evaluation {

  class HandCraftEvaluationPolicy {
    static constexpr int32_t kDoubledPawnPenalty = -15;
    static constexpr int32_t kIsolatedPawnPenalty = -15;
    static constexpr int32_t kOverloadedDefenderPenalty = -8;
    static constexpr int32_t kMinOverloadedDefenderPenalty = -16;
    static constexpr int32_t kMissingShelterPenalty = -8;
    static constexpr int32_t kSemiOpenKingFilePenalty = -6;
    static constexpr int32_t kOpenKingFilePenalty = -10;
    static constexpr int32_t kKingZoneAttackPenalty = -4;
    static constexpr int32_t kKingZoneDoubleAttackPenalty = -2;
    static constexpr int32_t kBishopPairBonus = 30;
    static constexpr int32_t kSemiOpenFileBonus = 10;
    static constexpr int32_t kOpenFileBonus = 20;
    static constexpr int32_t kMaxPhase = 24;
    static constexpr int32_t kMinKingWeaknessPenalty = -80;

    // Passed pawn bonus indexed by rank (relative to pawn advancement: rank 0..7)
    static constexpr std::array<int32_t, kSideSize> kPassedPawnBonusTable = { 0, 0, 5, 10, 20, 40, 80, 0 };

    // Indexed by piece type (0..5: Pawn, Knight, Bishop, Rook, Queen, King)
    static constexpr std::array<int32_t, kPieceSize> kPieceMobilityBonusTable = { 0, 4, 3, 2, 0, 0 };
    static constexpr std::array<int32_t, kPieceSize> kPieceMaterialValueTable = { 100, 320, 335, 500, 900, 0 };

    // Unified piece-square tables for 6 piece types, oriented from White's perspective.
    // For Black, flip the rank with square ^ 56.
    static constexpr std::array<std::array<int32_t, kSquareSize>, kPieceSize> kPiecePositionalValueTable = { {
        { // Pawn (fallback/base)
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0
        },
        { // Knight
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
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
        { // King (base)
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0,
              0,  0,  0,  0,  0,  0,  0,  0
        }
    }};

    // [Game Stage 0: Early Game, 1: Late Game][Square 0..63]
    static constexpr int32_t kPawnPositionalValueTable[2][64] =
    {
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

    // [Game Stage 0: Early Game, 1: End Game][Square 0..63]
    static constexpr int32_t kKingPositionalValueTable[2][64] =
    {
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

    // Precompute the squares where an enemy pawn could stop each pawn from passing.
    static constexpr auto kPassedPawnMasks = [] {
      std::array<std::array<Bitboard, kSquareSize>, kColorSize> masks{};
      // Build a separate mask for every color and starting square.
      for (Color color = kWhite; color < kColorSize; ++color) {
        for (Square square = 0; square < kSquareSize; ++square) {
          const int rank = static_cast<int>(getSquareRank(square));
          const int file = static_cast<int>(getSquareFile(square));
          // White moves toward lower rank indices; Black moves toward higher ones.
          for (int targetRank = 0; targetRank < static_cast<int>(kSideSize); ++targetRank) {
            if (color == kWhite ? targetRank >= rank : targetRank <= rank) {
              continue;
            }
            // Enemy pawns on this file or an adjacent file can oppose the pawn.
            for (int targetFile = std::max(0, file - 1);
                 targetFile <= std::min(7, file + 1); ++targetFile) {
              masks[color][square] = setSquare(
                masks[color][square],
                rankFileToSquare(static_cast<Square>(targetRank), static_cast<Square>(targetFile)));
            }
          }
        }
      }
      return masks;
    }();

    // Track attacked squares, repeated attacks, and each non-pawn defender's attacks.
    struct AttackInfo {
      Bitboard once = 0;
      Bitboard twice = 0;
      std::array<Bitboard, kSquareSize> defenderAttacks;
      uint32_t defenderCount = 0;

      void add(Bitboard mask) noexcept {
        twice |= once & mask;
        once |= mask;
      }

      void addDefender(Bitboard mask) noexcept {
        defenderAttacks[defenderCount++] = mask;
        add(mask);
      }
    };

    template <Color color>
    static constexpr Square positionalSquare(Square square) noexcept {
      return color == kWhite ? square : square ^ 56;
    }

    static constexpr int32_t blend(int32_t early, int32_t late, int32_t phase) noexcept {
      return (early * phase + late * (kMaxPhase - phase)) / kMaxPhase;
    }

    template <Color color, Piece piece>
    static int32_t scoreNonPawnPieces(const BoardState& state, Bitboard ownOccupancy,
                                      Bitboard bothOccupancy, Bitboard ownPawns,
                                      Bitboard enemyPawns, AttackInfo& attacks) noexcept {
      int32_t score = 0;
      for (Bitboard pieces = state.getPieces(color, piece); pieces; pieces = popPiece(pieces)) {
        const Square square = peekPiece(pieces);
        // Material and piece-square placement.
        score += kPieceMaterialValueTable[piece] +
          kPiecePositionalValueTable[piece][positionalSquare<color>(square)];

        Bitboard attackMask;
        if constexpr (piece == kKnight) {
          attackMask = getAttack<kKnight>(square);
        } else {
          attackMask = getAttack<piece>(square, bothOccupancy);
        }
        attacks.addDefender(attackMask);
        // Mobility.
        if constexpr (piece != kQueen) {
          score += static_cast<int32_t>(countPiece(attackMask & ~ownOccupancy)) *
            kPieceMobilityBonusTable[piece];
        }
        // Rook files.
        if constexpr (piece == kRook) {
          const Bitboard file = kSquareToFileMasks[square];
          if ((ownPawns & file) == 0) {
            score += (enemyPawns & file) == 0 ? kOpenFileBonus : kSemiOpenFileBonus;
          }
        }
      }
      return score;
    }

    // Score one side and collect its attacks for the later overload and king checks.
    template <Color color>
    static int32_t scoreSide(const BoardState& state, Bitboard ownOccupancy,
                             Bitboard bothOccupancy, int32_t phase,
                             AttackInfo& attacks) noexcept {
      constexpr Color enemy = getOtherColor(color);
      const Bitboard pawns = state.getPieces(color, kPawn);
      const Bitboard enemyPawns = state.getPieces(enemy, kPawn);
      int32_t score = 0;

      for (Bitboard pieces = pawns; pieces; pieces = popPiece(pieces)) {
        const Square square = peekPiece(pieces);
        const Square tableSquare = positionalSquare<color>(square);
        // Material and phase-blended pawn piece-square placement.
        score += kPieceMaterialValueTable[kPawn] +
          blend(kPawnPositionalValueTable[0][tableSquare],
                kPawnPositionalValueTable[1][tableSquare], phase);

        const Bitboard file = kSquareToFileMasks[square];
        const Bitboard adjacentFiles = shiftLeft(file) | shiftRight(file);
        // Isolated pawns.
        if ((pawns & adjacentFiles) == 0) {
          score += kIsolatedPawnPenalty;
        }
        // Passed pawns.
        if ((enemyPawns & kPassedPawnMasks[color][square]) == 0) {
          const Square advancement = color == kWhite
            ? 7 - getSquareRank(square) : getSquareRank(square);
          score += kPassedPawnBonusTable[advancement];
        }
        attacks.add(getAttack<kPawn, color>(square));
      }

      // Doubled pawns.
      for (Square file = 0; file < kSideSize; ++file) {
        const int32_t count = static_cast<int32_t>(countPiece(pawns & (kFileAMask << file)));
        if (count > 1) {
          score += (count - 1) * kDoubledPawnPenalty;
        }
      }

      score += scoreNonPawnPieces<color, kKnight>(
        state, ownOccupancy, bothOccupancy, pawns, enemyPawns, attacks);
      score += scoreNonPawnPieces<color, kBishop>(
        state, ownOccupancy, bothOccupancy, pawns, enemyPawns, attacks);
      score += scoreNonPawnPieces<color, kRook>(
        state, ownOccupancy, bothOccupancy, pawns, enemyPawns, attacks);
      score += scoreNonPawnPieces<color, kQueen>(
        state, ownOccupancy, bothOccupancy, pawns, enemyPawns, attacks);

      // Bishop pair.
      if (countPiece(state.getPieces(color, kBishop)) >= 2) {
        score += kBishopPairBonus;
      }
      const Square kingSquare = peekPiece(state.getPieces(color, kKing));
      // King piece-square placement, blended by game phase.
      score += blend(kKingPositionalValueTable[0][positionalSquare<color>(kingSquare)],
                     kKingPositionalValueTable[1][positionalSquare<color>(kingSquare)], phase);
      attacks.add(getAttack<kKing>(kingSquare));
      return score;
    }

    // Overloaded defenders: penalize a piece that alone protects multiple attacked valuable pieces.
    template <Color color>
    static int32_t overloadedDefenderPenalty(const BoardState& state,
                                              const AttackInfo& own,
                                              const AttackInfo& enemy) noexcept {
      const Bitboard valuable = state.getPieces(color, kKnight) |
        state.getPieces(color, kBishop) | state.getPieces(color, kRook) |
        state.getPieces(color, kQueen);
      const Bitboard soleDefenderTargets = valuable & enemy.once & ~own.twice;
      int32_t penalty = 0;
      for (uint32_t i = 0; i < own.defenderCount; ++i) {
        const int32_t targets = static_cast<int32_t>(
          countPiece(own.defenderAttacks[i] & soleDefenderTargets));
        if (targets > 1) {
          penalty += std::max(kMinOverloadedDefenderPenalty,
                              kOverloadedDefenderPenalty * (targets - 1));
        }
      }
      return penalty;
    }

    // Combine pawn shelter, open files, and nearby attacks, then scale by game phase.
    template <Color color>
    static int32_t kingWeakness(const BoardState& state, Bitboard ownPawns,
                                Bitboard enemyPawns, const AttackInfo& enemyAttacks,
                                int32_t phase) noexcept {
      if (phase == 0) {
        return 0;
      }
      const Square kingSquare = peekPiece(state.getPieces(color, kKing));
      const int kingRank = static_cast<int>(getSquareRank(kingSquare));
      const int kingFile = static_cast<int>(getSquareFile(kingSquare));
      int32_t penalty = 0;
      // Check the king's file and its neighbors for pawn cover and open files.
      for (int file = std::max(0, kingFile - 1); file <= std::min(7, kingFile + 1); ++file) {
        Bitboard shelter = 0;
        // The shelter is the first two squares ahead of the king on this file.
        for (int distance = 1; distance <= 2; ++distance) {
          const int rank = color == kWhite ? kingRank - distance : kingRank + distance;
          if (rank >= 0 && rank < static_cast<int>(kSideSize)) {
            shelter = setSquare(
              shelter, rankFileToSquare(static_cast<Square>(rank), static_cast<Square>(file)));
          }
        }
        // King pawn shelter.
        if ((ownPawns & shelter) == 0) {
          penalty += kMissingShelterPenalty;
        }
        // Open files near the king.
        const Bitboard fileMask = kFileAMask << file;
        if ((ownPawns & fileMask) == 0) {
          penalty += (enemyPawns & fileMask) == 0
            ? kOpenKingFilePenalty : kSemiOpenKingFilePenalty;
        }
      }

      // Attack pressure near the king.
      const Bitboard kingZone = getAttack<kKing>(kingSquare) | toBitboard(kingSquare);
      penalty += kKingZoneAttackPenalty * static_cast<int32_t>(countPiece(enemyAttacks.once & kingZone));
      penalty += kKingZoneDoubleAttackPenalty * static_cast<int32_t>(countPiece(enemyAttacks.twice & kingZone));
      return std::max(penalty, kMinKingWeaknessPenalty) * phase / kMaxPhase;
    }

  public:
    int32_t evaluate(const BoardState& state) const noexcept {
      // Game phase: weight remaining minor and major pieces on a 24-point scale.
      const int32_t phase = std::min(kMaxPhase,
        static_cast<int32_t>(
          countPiece(state.getPieces(kWhite, kKnight) | state.getPieces(kBlack, kKnight)) +
          countPiece(state.getPieces(kWhite, kBishop) | state.getPieces(kBlack, kBishop)) +
          2 * countPiece(state.getPieces(kWhite, kRook) | state.getPieces(kBlack, kRook)) +
          4 * countPiece(state.getPieces(kWhite, kQueen) | state.getPieces(kBlack, kQueen))));
      // Basic dead positions: bare kings or a lone minor piece against a bare king.
      if (phase <= 1 &&
          (state.getPieces(kWhite, kPawn) | state.getPieces(kBlack, kPawn)) == 0) {
        return 0;
      }

      const Bitboard whiteOccupancy = state.getOccupancy(kWhite);
      const Bitboard blackOccupancy = state.getOccupancy(kBlack);
      const Bitboard bothOccupancy = whiteOccupancy | blackOccupancy;

      // Both attack maps are needed for overload and king weakness penalties.
      AttackInfo whiteAttacks;
      AttackInfo blackAttacks;
      int32_t whiteScore = scoreSide<kWhite>(state, whiteOccupancy, bothOccupancy, phase, whiteAttacks);
      int32_t blackScore = scoreSide<kBlack>(state, blackOccupancy, bothOccupancy, phase, blackAttacks);
      whiteScore += overloadedDefenderPenalty<kWhite>(state, whiteAttacks, blackAttacks);
      blackScore += overloadedDefenderPenalty<kBlack>(state, blackAttacks, whiteAttacks);
      whiteScore += kingWeakness<kWhite>(
        state, state.getPieces(kWhite, kPawn), state.getPieces(kBlack, kPawn), blackAttacks, phase);
      blackScore += kingWeakness<kBlack>(
        state, state.getPieces(kBlack, kPawn), state.getPieces(kWhite, kPawn), whiteAttacks, phase);
      // Negamax expects the score from the side to move's perspective.
      const int32_t netScore = whiteScore - blackScore;
      return state.getColorToMove() == kWhite ? netScore : -netScore;
    }
  };
}
