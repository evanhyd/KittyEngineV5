#pragma once
#include "boardstate.h"
#include "searching_policy.h"
#include "time_control_policy.h"
#include "notation.h"
#include "position_fens.h"
#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <format>
#include <optional>
#include <span>
#include <vector>

namespace bb {
  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  class Board {
    SearchPolicy searchingPolicy_;
    TimePolicy timeControlPolicy_;
    BoardState state_;
    std::vector<Move> pastMoves_;
    searching::PVLine pvLine_;

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
      pastMoves_.reserve(256);
    }

    constexpr BoardState& getState() {
      return state_;
    }

    constexpr const BoardState& getState() const {
      return state_;
    }

    constexpr void setPosition(std::string_view fen) {
      // TODO: reset search policy state if needed.
      state_.setPosition(fen);
      pastMoves_.clear();
      pvLine_.clear();
    }

    void setPosition(std::string_view fen, std::span<const std::string_view> moves) {
      const BoardState previousState = state_;
      const auto matchesPosition = [](const BoardState& lhs, const BoardState& rhs) {
        return lhs.getHash() == rhs.getHash() &&
          lhs.getHalfmoveClock() == rhs.getHalfmoveClock() &&
          lhs.getFullmoveNumber() == rhs.getFullmoveNumber();
      };
      const searching::PVLine previousPV = pvLine_;
      std::vector<Move> previousMoves = std::move(pastMoves_);
      try {
        setPosition(fen);
        bool reachedPreviousPosition = false;
        const auto restorePreviousPV = [&] {
          if (!reachedPreviousPosition && matchesPosition(state_, previousState)) {
            pvLine_ = previousPV;
            reachedPreviousPosition = true;
          }
        };
        restorePreviousPV();
        for (const std::string_view moveText : moves) {
          playMove(moveText);
          restorePreviousPV();
        }

        if (!reachedPreviousPosition && !previousPV.empty()) {
          const auto matchesExpectedSuccessor = [&]<Side ally>() {
            MoveList legalMoves;
            previousState.generateMoves<ally>(legalMoves);
            const Move pvMove = previousPV.front();
            if (std::find(legalMoves.begin(), legalMoves.end(), pvMove) == legalMoves.end()) {
              return false;
            }
            BoardState nextState = previousState;
            nextState.makeMove<ally>(pvMove);
            return matchesPosition(nextState, state_);
          };
          const bool matches = previousState.getSideToMove() == White
            ? matchesExpectedSuccessor.template operator()<White>()
            : matchesExpectedSuccessor.template operator()<Black>();
          if (matches) {
            pvLine_ = previousPV;
            advancePV(previousPV.front());
          }
        }
      } catch (...) {
        state_ = previousState;
        pastMoves_ = std::move(previousMoves);
        pvLine_ = previousPV;
        throw;
      }
    }

    searching::SearchResult search(int maxDepth, const std::optional<time_control::TimeControl>& timeControl, auto resultCallback) {
      timeControlPolicy_.set(timeControl);
      maxDepth = std::min(maxDepth, searching::kMaxDepthHardCutoff);

      const auto iterativeDeepening = [&]<Side ally>() {
        searching::SearchResult result{
          .score = 0
        };

        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{
            .maxDepth = depth,
            .pastEval = result.score,
            .pvLine = pvLine_,
          };
          result = searchingPolicy_.template search<ally>(state_, param);
          pvLine_ = result.pvLine;
          resultCallback(result);

          if (!timeControlPolicy_.shouldContinue(ally)) {
            break;
          }
        }

        return result;
      };

      if (state_.getSideToMove() == White) {
        return iterativeDeepening.template operator()<White>();
      } else {
        return iterativeDeepening.template operator()<Black>();
      }
    }

    constexpr void playMove(std::string_view moveText) {
      const auto playMoveImpl = [&]<Side ally>() {
        MoveList moves;
        state_.generateMoves<ally>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<ally>(move);
            pastMoves_.push_back(move);
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
  };

  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  Board(SearchPolicy, TimePolicy) -> Board<SearchPolicy, TimePolicy>;
}
