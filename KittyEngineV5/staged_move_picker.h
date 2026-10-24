#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include "node_meta.h"
#include <algorithm>
#include <cstdint>
#include <utility>

namespace bb::searching {

  // Pick moves for diff stages.
  class StagedMovePicker {
    enum class Stage {
      TTMove,
      UnsortedViolentMove,
      SortedViolentMove,
      UnsortedNonViolentMove,
      SortedNonViolentMove,
    };

    // Move Ordering Priorities Magic.
    static constexpr int32_t kPromotionPriority = 1'000'000;
    static constexpr int32_t kUnderpromotionPriority = -2'000'000;
    static constexpr int32_t kViolentMovePriority = 100'000;
    static constexpr std::array<int32_t, kPieceSize> kVictimPriority{ 1100, 1320, 1330, 1500, 1900, 5000 };
    static constexpr int32_t kKillerMovePriority = 100;
    static constexpr int32_t kCastlingPriority = 30;

    static constexpr int32_t capturePriority(Piece attacker, Piece victim) noexcept {
      return kVictimPriority[victim] - attacker;
    }

    template <Side ally>
    int32_t staticExchangeEvaluate(const BoardState& state, Move move) const noexcept {
      constexpr Side enemy = getOtherSide(ally);
      const Square square = move.to();
      Bitboard occupancy = moveSquare(state.getOccupancy(White) | state.getOccupancy(Black), move.from(), square);
      if (move.isEnpassant()) {
        occupancy = unsetSquare(occupancy, ally == White ? squareDown(square) : squareUp(square));
      }
      const Piece promotion = move.getPromotedPieceType();
      const Piece attacker = promotion == NoPiece ? move.getMovedPiece() : promotion;
      const Piece victim = move.isCapture() ? state.getPieceAt<enemy>(square) : (move.isEnpassant() ? Pawn : NoPiece);
      std::array<Piece, kSideSize> pieceIndex{ Pawn, Pawn };

      auto popLVA = [&](Side side) -> Piece {
        for (Piece piece = pieceIndex[side]; piece < kPieceSize; ++piece) {
          Bitboard attackers = 0;
          switch (piece) {
          case Pawn: attackers = (side == White ? getAttack<Pawn, Black>(square) : getAttack<Pawn, White>(square)) & state.getPieces(side, Pawn); break;
          case Knight: attackers = getAttack<Knight>(square) & state.getPieces(side, Knight); break;
          case Bishop: attackers = getAttack<Bishop>(square, occupancy) & state.getPieces(side, Bishop); break;
          case Rook: attackers = getAttack<Rook>(square, occupancy) & state.getPieces(side, Rook); break;
          case Queen: attackers = getAttack<Queen>(square, occupancy) & state.getPieces(side, Queen); break;
          case King: attackers = getAttack<King>(square) & state.getPieces(side, King); break;
          default: break;
          }
          attackers &= occupancy;
          if (attackers) {
            occupancy = unsetSquare(occupancy, peekPiece(attackers));
            return piece;
          }
          // Pawn and knight attacks cannot be uncovered by removing blockers.
          if (piece < Bishop) {
            pieceIndex[side] = piece + 1;
          }
        }
        return NoPiece;
      };

      // https://chess.stackexchange.com/questions/44853/whats-the-maximum-number-of-pieces-attacking-a-single-square
      SmallVec<int32_t, 32> gains;
      gains.push(victim == NoPiece ? 0 : capturePriority(attacker, victim));
      Piece exchangedPiece = attacker;
      Side side = enemy;

      while (gains.size() < gains.capacity()) {
        const Piece nextAttacker = popLVA(side);
        if (nextAttacker == NoPiece) {
          break;
        }
        gains.push(capturePriority(nextAttacker, exchangedPiece));
        exchangedPiece = nextAttacker;
        side = getOtherSide(side);
      }

      int32_t reply = 0;
      for (size_t i = gains.size() - 1; i > 0; --i) {
        reply = std::max(0, gains[i] - reply); // Each side may decline to recapture.
      }
      return gains.front() - reply;
    }

    template <NodeMeta meta>
    void sortMoves(auto begin, auto end, const BoardState& state, const auto& killerMoves, const auto& historyTable) noexcept {
      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      static SmallVec<PriorityMove> scoredMoves{};
      scoredMoves.resize(end - begin);

      // Calculate the priority.
      std::transform(begin, end, scoredMoves.begin(), [&](const Move& move) {
        int32_t priority = 0;

        if (move.isViolentMove()) {
          priority += kViolentMovePriority + staticExchangeEvaluate<meta.side>(state, move);
          if (const Piece promotion = move.getPromotedPieceType(); promotion != NoPiece) {
            priority += (promotion == Queen ? kPromotionPriority : kUnderpromotionPriority);
          }
        } else if constexpr (meta.isInternal()) {
          // Killer move heuristics.
          for (size_t i = 0; i < killerMoves.size() && killerMoves[i]; ++i) {
            if (move == killerMoves[i]) {
              priority += kKillerMovePriority - int32_t(i);
            }
          }

          // History heuristics.
          priority += historyTable[meta.side][move.from()][move.to()];

          if (move.isCastling()) {
            priority += kCastlingPriority;
          }
        }

        return PriorityMove{ priority, move };
      });

      // Sort the moves based on priority, then map back to the original moves list.
      std::ranges::sort(scoredMoves, std::greater<>{}, & PriorityMove::priority);
      std::ranges::transform(scoredMoves, begin, &PriorityMove::move);
    }

    Stage stage = Stage::TTMove;
    uint32_t index = 0;

  public:
    template <NodeMeta meta>
    Move pick(MoveList& moves, Move ttMove, const BoardState& state, const auto& killerMoves, const auto& historyTable) noexcept {
      switch (stage) {
      case Stage::TTMove: {
        stage = Stage::UnsortedViolentMove;
        if constexpr (meta.isInternal()) {
          if (ttMove) {
            auto it = std::find(moves.begin(), moves.end(), ttMove);
            std::swap(moves[index], *it);
            ++index;
            return ttMove;
          }
        }
        [[fallthrough]];
      }
      case Stage::UnsortedViolentMove: {
        int endIndex = index;
        for (int i = index; i < moves.size(); ++i) {
          if (moves[i].isViolentMove()) {
            std::swap(moves[endIndex], moves[i]);
            ++endIndex;
          }
        }
        sortMoves<meta>(moves.begin() + index, moves.begin() + endIndex, state, killerMoves, historyTable);
        stage = Stage::SortedViolentMove;
        [[fallthrough]];
      }
      case Stage::SortedViolentMove: {
        if (moves[index].isViolentMove()) {
          return moves[index++];
        }
        stage = Stage::UnsortedNonViolentMove;
        [[fallthrough]];
      }
      case Stage::UnsortedNonViolentMove: {
        sortMoves<meta>(moves.begin() + index, moves.end(), state, killerMoves, historyTable);
        stage = Stage::SortedNonViolentMove;
        [[fallthrough]];
      }
      case Stage::SortedNonViolentMove: [[fallthrough]];
      default: {
        return moves[index++];
      }
      }
    }
  };
}
