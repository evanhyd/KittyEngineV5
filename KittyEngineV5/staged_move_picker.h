#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include "node_meta.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <utility>

namespace bb::searching {

  // Lazily pick move by reducing sorting overhead.
  // Queen Promotions -> Good Captures -> Killer Moves -> Quiet Moves -> Bad Violent Moves -> Underpromotions.
  class StagedMovePicker {
    enum class Stage : uint8_t {
      TTMove,
      PreSort,
      Sort,
    };

    static constexpr int32_t kSeeMargin = 0;

    // Move Ordering Priorities.
    static constexpr int32_t kPromotionPriority = 2'000'000;
    static constexpr int32_t kGoodCapturePriority = 1'000'000;
    static constexpr int32_t kKillerPriority = 10'000;
    static constexpr int32_t kBadViolentPriority = -1'000'000;
    static constexpr int32_t kUnderpromotionPriority = -2'000'000;
    static constexpr int32_t kCastlingPriority = 30;

    static constexpr std::array<int32_t, kPieceSize> kVictimPriority{ 1100, 1320, 1330, 1500, 1900, 5000 };

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
      const size_t count = static_cast<size_t>(end - begin);
      if (count == 0) {
        badViolentBegin_ = end;
        return;
      }

      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      static SmallVec<PriorityMove> scoredMoves{};
      scoredMoves.resize(count);

      size_t badViolentCount = 0;

      auto scoreViolentMove = [&](const Move& move) noexcept -> int32_t {
        const Piece promotion = move.getPromotedPieceType();
        if (promotion == Queen) {
          return kPromotionPriority + staticExchangeEvaluate<meta.side>(state, move);
        }
        if (promotion != NoPiece) {
          // Underpromotion is bad, avoid calculating SEE.
          ++badViolentCount;
          return kUnderpromotionPriority + static_cast<int32_t>(promotion);
        }
        const int32_t see = staticExchangeEvaluate<meta.side>(state, move);
        if (see < kSeeMargin) {
          ++badViolentCount;
          return kBadViolentPriority + see;
        }
        return kGoodCapturePriority + see;
      };

      if constexpr (meta.isQuiescence()) {
        std::transform(begin, end, scoredMoves.begin(), [&](const Move& move) {
          return PriorityMove{ scoreViolentMove(move), move };
        });
      } else {
        std::transform(begin, end, scoredMoves.begin(), [&](const Move& move) {
          if (move.isViolentMove()) {
            return PriorityMove{ scoreViolentMove(move), move };
          }
          for (size_t k = 0; k < killerMoves.size() && killerMoves[k]; ++k) {
            if (move == killerMoves[k]) {
              return PriorityMove{ kKillerPriority - static_cast<int32_t>(k), move };
            }
          }
          int32_t priority = historyTable[meta.side][move.from()][move.to()];
          if (move.isCastling()) {
            priority += kCastlingPriority;
          }
          return PriorityMove{ priority, move };
        });
      }

      if (count > 1) {
        std::ranges::sort(scoredMoves, std::greater<>{}, &PriorityMove::priority);
        std::ranges::transform(scoredMoves, begin, &PriorityMove::move);
      }

      badViolentBegin_ = end - badViolentCount;
    }

    Stage stage = Stage::TTMove;
    MoveList::iterator current{};
    MoveList::iterator badViolentBegin_{};

  public:
    [[nodiscard]] bool isBadViolentMove() const noexcept {
      return current > badViolentBegin_;
    }

    template <NodeMeta meta>
    Move pick(MoveList& moves, Move ttMove, const BoardState& state, const auto& killerMoves, const auto& historyTable) noexcept {
      switch (stage) {
      case Stage::TTMove: {
        stage = Stage::PreSort;
        current = moves.begin();
        badViolentBegin_ = moves.end();
        if constexpr (meta.isInternal()) {
          if (ttMove) {
            auto it = std::find(moves.begin(), moves.end(), ttMove);
            assert(it != moves.end());
            std::swap(*current, *it);
            return *current++;
          }
        }
        [[fallthrough]];
      }
      case Stage::PreSort: {
        sortMoves<meta>(current, moves.end(), state, killerMoves, historyTable);
        stage = Stage::Sort;
        [[fallthrough]];
      }
      case Stage::Sort: [[fallthrough]];
      default: {
        assert(current != moves.end());
        return *current++;
      }
      }
    }
  };
}
