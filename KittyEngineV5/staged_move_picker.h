#pragma once
#include "bitboard.h"
#include "boardstate.h"
#include "node_meta.h"
#include "history_heuristics.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
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
    static constexpr int32_t kBadViolentPriority = -1'000'000;
    static constexpr int32_t kUnderpromotionPriority = -2'000'000;

    static constexpr std::array<int32_t, kPieceSize> kVictimPriority{ 1100, 1320, 1320, 1500, 1900, 5000 };


    template <Side ally>
    int32_t staticExchangeEvaluate(const BoardState& state, Move move) const noexcept {
      constexpr Side enemy = getOtherSide(ally);
      const Square square = move.to();
      const std::array<Bitboard, kSideSize> sideOccupancy = { state.getOccupancy(White), state.getOccupancy(Black) };
      Bitboard occupancy = moveSquare(sideOccupancy[White] | sideOccupancy[Black], move.from(), square);
      if (move.isEnpassant()) {
        occupancy = unsetSquare(occupancy, ally == White ? squareDown(square) : squareUp(square));
      }

      const Piece promotion = move.getPromotedPieceType();
      const Piece attacker = promotion == NoPiece ? move.getMovedPiece() : promotion;
      const Piece victim = move.isCapture() ? state.getPieceAt<enemy>(square) : (move.isEnpassant() ? Pawn : NoPiece);
      std::array<Piece, kSideSize> pieceIndex{ Pawn, Pawn };

      // Constant attacks on 'square' that do not change as occupancy changes
      const std::array<Bitboard, kSideSize> pawnAttacks = { getAttack<Pawn, Black>(square), getAttack<Pawn, White>(square) };
      const Bitboard knightAttacks = getAttack<Knight>(square);
      const Bitboard kingAttacks   = getAttack<King>(square);
      const std::array<Square, kSideSize> kingSq = { peekPiece(state.getPieces(White, King)), peekPiece(state.getPieces(Black, King)) };
      const std::array<Bitboard, kSideSize> pinned = { state.getPinnedMask<White>(kingSq[White], sideOccupancy), state.getPinnedMask<Black>(kingSq[Black], sideOccupancy) };

      auto popLVA = [&](Side side) -> Piece {
        // Cache slider ray lookups for this ply so Queen reuses Bishop and Rook rays
        std::optional<Bitboard> bAttacks;
        std::optional<Bitboard> rAttacks;
        auto getBishopAttacks = [&] {
          if (!bAttacks) bAttacks = getAttack<Bishop>(square, occupancy);
          return *bAttacks;
        };
        auto getRookAttacks = [&] {
          if (!rAttacks) rAttacks = getAttack<Rook>(square, occupancy);
          return *rAttacks;
        };

        for (Piece piece = pieceIndex[side]; piece < kPieceSize; ++piece) {
          Bitboard attackers = 0;
          switch (piece) {
          case Pawn:   attackers = pawnAttacks[side] & state.getPieces(side, Pawn); break;
          case Knight: attackers = knightAttacks & state.getPieces(side, Knight); break;
          case Bishop: attackers = getBishopAttacks() & state.getPieces(side, Bishop); break;
          case Rook:   attackers = getRookAttacks() & state.getPieces(side, Rook); break;
          case Queen:  attackers = (getBishopAttacks() | getRookAttacks()) & state.getPieces(side, Queen); break;
          case King:   attackers = kingAttacks & state.getPieces(side, King); break;
          default: break;
          }
          attackers &= occupancy;

          if (attackers & pinned[side]) {
            // Filter out pinned pieces.
            for (Bitboard bb = attackers & pinned[side]; bb; bb = popPiece(bb)) {
              const Square sq = peekPiece(bb);
              if (piece == Knight || !isSquareSet(kLineOfSightMasks[kingSq[side]][sq], square)) {
                attackers = unsetSquare(attackers, sq);
              }
            }
          }

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
      gains.push(victim == NoPiece ? 0 : kVictimPriority[victim]);
      Piece exchangedPiece = attacker;
      Side side = enemy;

      while (gains.size() < gains.capacity()) {
        const Piece nextAttacker = popLVA(side);
        if (nextAttacker == NoPiece) {
          break;
        }
        gains.push(kVictimPriority[exchangedPiece]);
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
    void sortMoves(auto begin, auto end, const BoardState& state) noexcept {
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

      auto scoreMove = [&](Move move) noexcept -> int32_t {
        if (move.isViolentMove()) {
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
          constexpr Side enemy = getOtherSide(meta.side);
          const Piece victim = move.isCapture() ? state.getPieceAt<enemy>(move.to()) : (move.isEnpassant() ? Pawn : NoPiece);
          const int32_t capScore = victim != NoPiece ? history_.getCaptureScore<meta.side>(move.getMovedPiece(), move.to(), victim) : 0;
          return kGoodCapturePriority + see + capScore;
        }

        if constexpr (meta.isQuiescence()) {
          return 0; // Quiet evasion
        } else {
          return history_.getQuietScore<meta.side>(ply_, move, prev1_, prev2_);
        }
      };

      std::transform(begin, end, scoredMoves.begin(), [&](Move move) {
        return PriorityMove{ scoreMove(move), move };
      });

      if (count > 1) {
        std::ranges::sort(scoredMoves, std::greater<>{}, &PriorityMove::priority);
        std::ranges::transform(scoredMoves, begin, &PriorityMove::move);
      }

      badViolentBegin_ = end - badViolentCount;
    }

    int ply_;
    Move prev1_{};
    Move prev2_{};
    const HistoryHeuristics& history_;
    Stage stage = Stage::TTMove;
    MoveList::iterator current{};
    MoveList::iterator badViolentBegin_{};

  public:
    explicit StagedMovePicker(int ply, Move prev1, Move prev2, const HistoryHeuristics& history) noexcept
      : ply_(ply), prev1_(prev1), prev2_(prev2), history_(history) {}

    [[nodiscard]] bool isBadViolentMove() const noexcept {
      return current > badViolentBegin_;
    }

    template <NodeMeta meta>
    Move pick(MoveList& moves, Move ttMove, const BoardState& state) noexcept {
      switch (stage) {
      case Stage::TTMove: {
        stage = Stage::PreSort;
        current = moves.begin();
        badViolentBegin_ = moves.end();
        if constexpr (meta.isInternal()) {
          // If ttMove is valid, then it guarantees exists in the move list.
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
        sortMoves<meta>(current, moves.end(), state);
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