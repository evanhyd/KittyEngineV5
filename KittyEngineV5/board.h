#pragma once
#include "boardstate.h"
#include "notation.h"
#include "position_fens.h"
#include "searching_policy.h"
#include "time_control_policy.h"
#include <algorithm>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace bb {
  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  class Board {
    SearchPolicy searchingPolicy_;
    TimePolicy timeControlPolicy_;
    BoardState state_;
    searching::PositionHistory positionHistory_;
    searching::MoveHistory moveHistory_;
    searching::PVLine pvLine_;

    struct PVReplayCallback {
      template <bool Add>
      void markPiece(Side, Piece, Square) const noexcept {}
      void markCastle(CastlePermission) const noexcept {}
      void markEnpassant(Square) const noexcept {}
    };

    void advancePV(const Move& move) {
      if (pvLine_.empty() || pvLine_.front() != move) {
        pvLine_.clear();
        return;
      }
      std::shift_left(pvLine_.begin(), pvLine_.end(), 1);
      pvLine_.pop();
    }

  public:
    explicit Board(SearchPolicy searchPolicy, TimePolicy timePolicy)
      : searchingPolicy_(std::move(searchPolicy)), timeControlPolicy_(std::move(timePolicy)) {
      setPosition(fen::kStartPosition);
    }

    constexpr BoardState& getState() {
      return state_;
    }

    constexpr const BoardState& getState() const {
      return state_;
    }

#if KITTY_ENABLE_SYZYGY
    bool hasTablebaseService() const noexcept {
      if constexpr (requires { searchingPolicy_.hasTablebaseService(); }) {
        return searchingPolicy_.hasTablebaseService();
      } else {
        return false;
      }
    }

    std::string setTablebaseOption(std::string_view name, std::string_view value) {
      if constexpr (requires { searchingPolicy_.setTablebaseOption(name, value); }) {
        pvLine_.clear();
        return searchingPolicy_.setTablebaseOption(name, value);
      } else {
        throw std::invalid_argument("Search policy does not support Syzygy");
      }
    }
#endif

    void reset() {
      searchingPolicy_.reset();
      positionHistory_.clear();
      positionHistory_.push(state_.getRepetitionHash());
      moveHistory_.clear();
      pvLine_.clear();
    }

    void setPosition(std::string_view fen) {
      state_.setPosition(fen);
      searchingPolicy_.invalidateEvaluation();
      positionHistory_.clear();
      positionHistory_.push(state_.getRepetitionHash());
      moveHistory_.clear();
      pvLine_.clear();
    }

    void setPosition(std::string_view fen, std::span<const std::string_view> moves) {
      const BoardState previousState = state_;
      const searching::PositionHistory previousPositionHistory = positionHistory_;
      const searching::MoveHistory previousMoveHistory = moveHistory_;
      const searching::PVLine previousPV = pvLine_;
      try {
        setPosition(fen);
        for (const std::string_view moveText : moves) {
          playMove(moveText);
        }

        // Keep the old PV if replay ends at the same position or at a position along that PV.
        if (state_.getHash() == previousState.getHash()) {
          pvLine_ = previousPV;
        } else if (!previousPV.empty()) {
          BoardState pvState = previousState;
          for (size_t i = 0; i < previousPV.size(); ++i) {
            MoveList legalMoves;
            const Side ally = pvState.getSideToMove();
            if (ally == White) {
              pvState.generateMoves<White>(legalMoves);
            } else {
              pvState.generateMoves<Black>(legalMoves);
            }
            if (std::find(legalMoves.begin(), legalMoves.end(), previousPV[i]) == legalMoves.end()) {
              break;
            }
            if (ally == White) {
              pvState.makeMove<White>(previousPV[i], PVReplayCallback{});
            } else {
              pvState.makeMove<Black>(previousPV[i], PVReplayCallback{});
            }
            if (pvState.getHash() == state_.getHash()) {
              pvLine_.resize(previousPV.size() - i - 1);
              std::copy(previousPV.begin() + i + 1, previousPV.end(), pvLine_.begin());
              break;
            }
          }
        }
      } catch (...) {
        state_ = previousState;
        positionHistory_ = previousPositionHistory;
        moveHistory_ = previousMoveHistory;
        pvLine_ = previousPV;
        throw;
      }
    }

    void playMove(std::string_view moveText) {
      const auto playMoveImpl = [&]<Side ally>() {
        MoveList moves;
        state_.generateMoves<ally>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<ally>(move, searchingPolicy_);
            positionHistory_.push(state_.getRepetitionHash());
            moveHistory_.push(move);
            advancePV(move);
            return;
          }
        }
        throw std::invalid_argument(std::format("Illegal UCI move: {}", moveText));
      };

      if (state_.getSideToMove() == White) {
        playMoveImpl.template operator()<White>();
      } else {
        playMoveImpl.template operator()<Black>();
      }
    }

    searching::SearchResult search(int maxDepth, const std::optional<time_control::TimeControl>& timeControl, auto resultCallback) {
      timeControlPolicy_.set(timeControl);
      maxDepth = std::min(maxDepth, searching::kMaxDepthHardCutoff);

      const auto iterativeDeepening = [&]<Side ally
#if KITTY_ENABLE_SYZYGY
        , bool UseTablebases = false
#endif
      >() {
        searching::SearchResult result{
          .score = 0,
          .pvLine = &pvLine_,
        };

        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{
            .maxDepth = depth,
            .pastEval = result.score,
            .positionHistory = positionHistory_,
            .pvLine = pvLine_,
          };
#if KITTY_ENABLE_SYZYGY
          if constexpr (UseTablebases) {
            // Timed searches request the hard depth limit, but often finish
            // before coverage is reachable. Dispatch once per iteration.
            // MSVC needs this guard for policies without Syzygy support.
            if constexpr (requires { searchingPolicy_.canProbeTablebases(state_, depth); }) {
              if (searchingPolicy_.canProbeTablebases(state_, depth)) {
                result = searchingPolicy_.template search<ally, true>(state_, param);
              } else {
                result = searchingPolicy_.template search<ally>(state_, param);
              }
            }
          }
          else
#endif
          {
            result = searchingPolicy_.template search<ally>(state_, param);
          }
          resultCallback(result);

          if (!timeControlPolicy_.shouldContinue(ally)) {
            break;
          }
        }

        return result;
      };

#if KITTY_ENABLE_SYZYGY
      // Entirely unreachable searches use the ordinary loop. Otherwise choose
      // the specialization at each iteration, never with a per-node toggle.
      if constexpr (requires { searchingPolicy_.canProbeTablebases(state_, maxDepth); searchingPolicy_.beginTablebaseSearch(); }) {
        if (searchingPolicy_.canProbeTablebases(state_, maxDepth)) {
          searchingPolicy_.beginTablebaseSearch();
          if (state_.getSideToMove() == White) {
            return iterativeDeepening.template operator()<White, true>();
          }
          return iterativeDeepening.template operator()<Black, true>();
        }
      }
#endif
      if (state_.getSideToMove() == White) {
        return iterativeDeepening.template operator()<White>();
      } else {
        return iterativeDeepening.template operator()<Black>();
      }
    }
  };

  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  Board(SearchPolicy, TimePolicy) -> Board<SearchPolicy, TimePolicy>;
}
