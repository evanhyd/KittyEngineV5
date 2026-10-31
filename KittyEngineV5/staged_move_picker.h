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

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 26495) // scores_ elements are intentionally uninitialized until written
#endif

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

    static constexpr std::array<int32_t, kPieceSize> kVictimPriority{ 100, 320, 320, 500, 900, 10000 };

    struct ScoreEntry {
      int32_t score;
      int16_t see;
    };

    template <NodeMeta meta>
    int32_t scoreViolent(const BoardState& state, Move m, int32_t see) const noexcept {
      const Piece promo = m.getPromotedPieceType();
      if (promo == Queen) {
        return kPromotionPriority + see;
      }
      if (promo != NoPiece) {
        return kUnderpromotionPriority + static_cast<int32_t>(promo);
      }
      if (see < kSeeMargin) {
        return kBadViolentPriority + see;
      }
      constexpr Side enemy = getOtherSide(meta.side);
      const Piece victim = m.isCapture() ? state.getPieceAt<enemy>(m.to()) : (m.isEnpassant() ? Pawn : NoPiece);
      const int32_t capScore = victim != NoPiece ? historyHeuristics_.getCaptureScore<meta.side>(m.getMovedPiece(), m.to(), victim) : 0;
      return kGoodCapturePriority + see + capScore;
    }

    int ply_;
    Move prev1_{};
    Move prev2_{};
    const HistoryHeuristics& historyHeuristics_;

    Stage stage_ = Stage::TTMove;

    size_t current_ = 0;
    size_t goodViolentEnd_ = 0;
    size_t badQuietBegin_ = 0;
    size_t badViolentBegin_ = 0;

    MoveList moves_{};
    ScoreEntry scores_[MoveList{}.capacity()];

  public:
    struct PickResult {
      Move move;
      int32_t see = 0;

      constexpr operator Move() const noexcept { return move; }
    };

    template <Side ally>
    static int32_t staticExchangeEvaluate(const BoardState& state, Move move) noexcept {
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

    explicit StagedMovePicker(int ply, Move prev1, Move prev2, const HistoryHeuristics& history) noexcept
      : ply_(ply), prev1_(prev1), prev2_(prev2), historyHeuristics_(history) {}

    MoveList& getMoveBuffer() noexcept { return moves_; }

    const MoveList& getMoveBuffer() const noexcept { return moves_; }

    size_t size() const noexcept { return moves_.size(); }

    bool empty() const noexcept { return moves_.empty(); }

    void rePick() noexcept {
      stage_ = Stage::TTMove;
      current_ = 0;
      goodViolentEnd_ = 0;
      badQuietBegin_ = moves_.size();
      badViolentBegin_ = moves_.size();
    }

    void clear() noexcept {
      moves_.clear();
      rePick();
    }

    bool isBadViolentMove() const noexcept {
      return current_ > badViolentBegin_;
    }

    template <NodeMeta meta>
    PickResult pick(Move ttMove, const BoardState& state) noexcept {
      assert(current_ < moves_.size());
      auto pickKiller = [&](Move killer) noexcept -> bool {
        if (killer) {
          for (size_t i = current_; i < badViolentBegin_; ++i) {
            if (moves_[i] == killer) {
              std::swap(moves_[current_], moves_[i]);
              return true;
            }
          }
        }
        return false;
      };

      while (true) {
        switch (stage_) {
        case Stage::TTMove: {
          stage_ = Stage::GoodViolent;
          current_ = 0;
          goodViolentEnd_ = 0;
          badQuietBegin_ = moves_.size();
          badViolentBegin_ = moves_.size();
          if constexpr (meta.isInternal()) {
            // The TT move is guaranteed to exist in the move list if ttMove is non-zero.
            if (ttMove) {
              auto it = std::find(moves_.begin(), moves_.end(), ttMove);
              assert(it != moves_.end());
              std::swap(moves_[0], *it);
              const int32_t see = ttMove.isViolentMove() ? staticExchangeEvaluate<meta.side>(state, ttMove) : 0;
              const size_t idx = current_++;
              return { moves_[idx], see };
            }
          }
          [[fallthrough]];
        }

        case Stage::GoodViolent: {
          stage_ = Stage::YieldGoodViolent;
          const size_t totalMoves = moves_.size();
          if (current_ < totalMoves) {
            size_t front = current_;
            size_t scan = current_;
            size_t badViolentBegin = totalMoves;

            while (scan < badViolentBegin) {
              const Move m = moves_[scan];
              if (m.isViolentMove()) {
                const int32_t see = staticExchangeEvaluate<meta.side>(state, m);
                const int32_t score = scoreViolent<meta>(state, m, see);
                if (score >= 0) {
                  moves_[scan] = moves_[front];
                  moves_[front] = m;
                  scores_[front] = { score, static_cast<int16_t>(see) };
                  ++front;
                  ++scan;
                } else {
                  --badViolentBegin;
                  moves_[scan] = moves_[badViolentBegin];
                  moves_[badViolentBegin] = m;
                  scores_[badViolentBegin] = { score, static_cast<int16_t>(see) };
                }
              } else {
                ++scan;
              }
            }

            goodViolentEnd_ = front;
            badViolentBegin_ = badViolentBegin;
            badQuietBegin_ = badViolentBegin_;

            // Sort good violent moves [current_, goodViolentEnd_) by score descending.
            for (size_t i = current_ + 1; i < goodViolentEnd_; ++i) {
              const Move keyMove = moves_[i];
              const auto keyScore = scores_[i];
              size_t j = i;
              while (j > current_ && scores_[j - 1].score < keyScore.score) {
                moves_[j] = moves_[j - 1];
                scores_[j] = scores_[j - 1];
                --j;
              }
              moves_[j] = keyMove;
              scores_[j] = keyScore;
            }

            // Sort bad violent moves [badViolentBegin_, totalMoves) by score descending.
            for (size_t i = badViolentBegin_ + 1; i < totalMoves; ++i) {
              const Move keyMove = moves_[i];
              const auto keyScore = scores_[i];
              size_t j = i;
              while (j > badViolentBegin_ && scores_[j - 1].score < keyScore.score) {
                moves_[j] = moves_[j - 1];
                scores_[j] = scores_[j - 1];
                --j;
              }
              moves_[j] = keyMove;
              scores_[j] = keyScore;
            }
          } else {
            goodViolentEnd_ = current_;
            badViolentBegin_ = current_;
            badQuietBegin_ = current_;
          }
          [[fallthrough]];
        }

        case Stage::YieldGoodViolent: {
          if (current_ < goodViolentEnd_) {
            const size_t idx = current_++;
            return { moves_[idx], scores_[idx].see };
          }
          stage_ = Stage::Killer1;
          [[fallthrough]];
        }

        case Stage::Killer1: {
          stage_ = Stage::Killer2;
          if constexpr (meta.isInternal()) {
            if (pickKiller(historyHeuristics_.getKiller(ply_, 0))) {
              const size_t idx = current_++;
              return { moves_[idx], 0 };
            }
          }
          [[fallthrough]];
        }

        case Stage::Killer2: {
          stage_ = Stage::Quiets;
          if constexpr (meta.isInternal()) {
            const Move k1 = historyHeuristics_.getKiller(ply_, 0);
            const Move k2 = historyHeuristics_.getKiller(ply_, 1);
            if (k2 != k1 && pickKiller(k2)) {
              const size_t idx = current_++;
              return { moves_[idx], 0 };
            }
          }
          [[fallthrough]];
        }

        case Stage::Quiets: {
          stage_ = Stage::YieldQuiets;
          if constexpr (meta.isInternal()) {
            const size_t n = badViolentBegin_ - current_;
            if (n > 1) {
              for (size_t i = current_; i < badViolentBegin_; ++i) {
                scores_[i].score = historyHeuristics_.getQuietScore<meta.side>(moves_[i], prev1_, prev2_);
              }

              for (size_t i = current_ + 1; i < badViolentBegin_; ++i) {
                const Move keyMove = moves_[i];
                const int32_t keyScore = scores_[i].score;
                size_t j = i;
                while (j > current_ && scores_[j - 1].score < keyScore) {
                  moves_[j] = moves_[j - 1];
                  scores_[j].score = scores_[j - 1].score;
                  --j;
                }
                moves_[j] = keyMove;
                scores_[j].score = keyScore;
              }

              badQuietBegin_ = current_;
              while (badQuietBegin_ < badViolentBegin_ && scores_[badQuietBegin_].score >= 0) {
                ++badQuietBegin_;
              }
            } else if (n == 1) {
              const int32_t s = historyHeuristics_.getQuietScore<meta.side>(moves_[current_], prev1_, prev2_);
              scores_[current_].score = s;
              badQuietBegin_ = (s < 0) ? current_ : badViolentBegin_;
            } else {
              badQuietBegin_ = current_;
            }
          }
          [[fallthrough]];
        }

        case Stage::YieldQuiets: {
          if (current_ < badViolentBegin_) {
            const size_t idx = current_++;
            return { moves_[idx], 0 };
          }
          stage_ = Stage::BadViolent;
          [[fallthrough]];
        }

        case Stage::BadViolent: {
          assert(current_ < moves_.size());
          const size_t idx = current_++;
          return { moves_[idx], scores_[idx].see };
        }
        default:
          assert(false);
#if defined(_MSC_VER)
          __assume(false);
#elif defined(__GNUC__) || defined(__clang__)
          __builtin_unreachable();
#endif
        }
      }
    }
  };
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
