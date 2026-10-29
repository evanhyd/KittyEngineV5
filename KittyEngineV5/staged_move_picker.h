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

namespace bb::searching {

  // Lazily pick moves in priority stages entirely in-place:
  // TT Move -> Good Violent (Queen promos & SEE >= 0) -> Killers -> Good Quiets -> Bad Quiets -> Bad Violent (SEE < 0 & Underpromos).
  class StagedMovePicker {
    enum class Stage : uint8_t {
      TTMove,
      GoodViolent,
      YieldGoodViolent,
      Killer1,
      Killer2,
      Quiets,
      YieldQuiets,
      BadViolent
    };

    static constexpr int32_t kSeeMargin = 0;
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

    template <typename It, typename Pred>
    static It forwardPartition(It first, It last, Pred pred) noexcept {
      first = std::find_if_not(first, last, pred);
      for (auto it = first; it != last; ++it) {
        if (pred(*it)) {
          std::iter_swap(first++, it);
        }
      }
      return first;
    }

    template <typename It, typename ScoreFn>
    static It sortAndSplitAtZero(It begin, It end, ScoreFn scoreFn) noexcept {
      const size_t n = static_cast<size_t>(end - begin);
      if (n <= 1) {
        return (n == 1 && scoreFn(*begin) < 0) ? begin : end;
      }

      int32_t scores[256];
      assert(n <= 256);
      for (size_t i = 0; i < n; ++i) {
        scores[i] = scoreFn(*(begin + i));
      }

      for (size_t i = 1; i < n; ++i) {
        const Move keyMove = *(begin + i);
        const int32_t keyScore = scores[i];
        size_t j = i;
        while (j > 0 && scores[j - 1] < keyScore) {
          *(begin + j) = *(begin + j - 1);
          scores[j] = scores[j - 1];
          --j;
        }
        *(begin + j) = keyMove;
        scores[j] = keyScore;
      }

      auto it = std::find_if(scores, scores + n, [](int32_t s) noexcept { return s < 0; });
      return begin + (it - scores);
    }

    template <NodeMeta meta>
    int32_t scoreViolent(const BoardState& state, Move m) const noexcept {
      const Piece promo = m.getPromotedPieceType();
      if (promo == Queen) {
        return kPromotionPriority + staticExchangeEvaluate<meta.side>(state, m);
      }
      if (promo != NoPiece) {
        return kUnderpromotionPriority + static_cast<int32_t>(promo);
      }
      const int32_t see = staticExchangeEvaluate<meta.side>(state, m);
      if (see < kSeeMargin) {
        return kBadViolentPriority + see;
      }
      constexpr Side enemy = getOtherSide(meta.side);
      const Piece victim = m.isCapture() ? state.getPieceAt<enemy>(m.to()) : (m.isEnpassant() ? Pawn : NoPiece);
      const int32_t capScore = victim != NoPiece ? history_.getCaptureScore<meta.side>(m.getMovedPiece(), m.to(), victim) : 0;
      return kGoodCapturePriority + see + capScore;
    }

    int ply_;
    Move prev1_{};
    Move prev2_{};
    const HistoryHeuristics& history_;

    Stage stage_ = Stage::TTMove;

    MoveList::iterator current_{};
    MoveList::iterator goodViolentEnd_{};
    MoveList::iterator badQuietBegin_{};
    MoveList::iterator badViolentBegin_{};

  public:
    explicit StagedMovePicker(int ply, Move prev1, Move prev2, const HistoryHeuristics& history) noexcept
      : ply_(ply), prev1_(prev1), prev2_(prev2), history_(history) {}

    bool isBadViolentMove() const noexcept {
      return current_ > badViolentBegin_;
    }

    bool isBadQuietMove() const noexcept {
      return current_ > badQuietBegin_ && current_ <= badViolentBegin_;
    }

    template <NodeMeta meta>
    Move pick(MoveList& moves, Move ttMove, const BoardState& state) noexcept {
      auto pickKiller = [&](Move killer) noexcept -> bool {
        if (killer) {
          auto it = std::find(current_, badViolentBegin_, killer);
          if (it != badViolentBegin_) {
            std::swap(*current_, *it);
            return true;
          }
        }
        return false;
      };

      while (true) {
        switch (stage_) {
        case Stage::TTMove: {
          stage_ = Stage::GoodViolent;
          current_ = moves.begin();
          goodViolentEnd_ = moves.begin();
          badQuietBegin_ = moves.end();
          badViolentBegin_ = moves.end();
          if constexpr (meta.isInternal()) {
            // The TT move is guaranteed to exist in the move list if ttMove is non-zero.
            if (ttMove) {
              auto it = std::find(moves.begin(), moves.end(), ttMove);
              assert(it != moves.end());
              std::swap(*current_, *it);
              return *current_++;
            }
          }
          [[fallthrough]];
        }

        case Stage::GoodViolent: {
          stage_ = Stage::YieldGoodViolent;
          auto violentsEnd = forwardPartition(current_, moves.end(), [](Move m) noexcept {
            return m.isViolentMove();
          });
          goodViolentEnd_ = sortAndSplitAtZero(current_, violentsEnd, [&](Move m) noexcept {
            return scoreViolent<meta>(state, m);
          });
          const size_t numBadViolents = static_cast<size_t>(violentsEnd - goodViolentEnd_);
          if (numBadViolents > 0) {
            std::rotate(goodViolentEnd_, violentsEnd, moves.end());
          }
          badViolentBegin_ = moves.end() - numBadViolents;
          badQuietBegin_ = badViolentBegin_;
          [[fallthrough]];
        }

        case Stage::YieldGoodViolent: {
          if (current_ < goodViolentEnd_) {
            return *current_++;
          }
          stage_ = Stage::Killer1;
          [[fallthrough]];
        }

        case Stage::Killer1: {
          stage_ = Stage::Killer2;
          if constexpr (meta.isInternal()) {
            if (pickKiller(history_.getKiller(ply_, 0))) {
              return *current_++;
            }
          }
          [[fallthrough]];
        }

        case Stage::Killer2: {
          stage_ = Stage::Quiets;
          if constexpr (meta.isInternal()) {
            const Move k1 = history_.getKiller(ply_, 0);
            const Move k2 = history_.getKiller(ply_, 1);
            if (k2 != k1 && pickKiller(k2)) {
              return *current_++;
            }
          }
          [[fallthrough]];
        }

        case Stage::Quiets: {
          stage_ = Stage::YieldQuiets;
          if constexpr (meta.isInternal()) {
            badQuietBegin_ = sortAndSplitAtZero(current_, badViolentBegin_, [&](Move m) noexcept {
              return history_.getQuietScore<meta.side>(m, prev1_, prev2_);
            });
          }
          [[fallthrough]];
        }

        case Stage::YieldQuiets: {
          if (current_ < badViolentBegin_) {
            return *current_++;
          }
          stage_ = Stage::BadViolent;
          [[fallthrough]];
        }

        case Stage::BadViolent: {
          return *current_++;
        }
        default:
          return Move{};
        }
      }
    }
  };
}
