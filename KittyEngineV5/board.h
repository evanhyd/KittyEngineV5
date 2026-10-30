#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "notation.h"
#include "position_fens.h"
#include "searching_policy.h"
#include "time_control_policy.h"
#include <algorithm>
#include <cstdint>
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

    void reset() {
      searchingPolicy_.reset();
      positionHistory_.clear();
      positionHistory_.push(state_.getHash());
      moveHistory_.clear();
    }

    void setPosition(std::string_view fen) {
      state_.setPosition(fen);
      searchingPolicy_.invalidateEvaluation();
      positionHistory_.clear();
      positionHistory_.push(state_.getHash());
      moveHistory_.clear();
    }

    void setPosition(std::string_view fen, std::span<const std::string_view> moves) {
      const BoardState previousState = state_;
      const searching::PositionHistory previousPositionHistory = positionHistory_;
      const searching::MoveHistory previousMoveHistory = moveHistory_;
      try {
        setPosition(fen);
        for (const std::string_view moveText : moves) {
          playMove(moveText);
        }
      } catch (...) {
        state_ = previousState;
        positionHistory_ = previousPositionHistory;
        moveHistory_ = previousMoveHistory;
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
            positionHistory_.push(state_.getHash());
            moveHistory_.push(move);
            return;
          }
        }
        throw std::invalid_argument(std::format("Illegal UCI move: {}", moveText));
      };

      if (state_.getSide() == White) {
        playMoveImpl.template operator()<White>();
      } else {
        playMoveImpl.template operator()<Black>();
      }
    }

    searching::SearchResult search(int maxDepth, const std::optional<time_control::TimeControl>& timeControl, auto resultCallback) {
      timeControlPolicy_.set(timeControl);
      maxDepth = std::min(maxDepth, searching::kMaxDepthHardCutoff);

      const auto iterativeDeepening = [&]<Side ally>() {
        searching::SearchResult result{};

        std::optional<MoveList> rootMoves;

        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{
            .maxDepth = depth,
            .pastEval = result.score,
            .positionHistory = positionHistory_,
            .rootMoves = rootMoves,
          };
          result = searchingPolicy_.template search<ally>(state_, param);
          resultCallback(result);

          constexpr int32_t mateScore = evaluation::kCheckmateWinScore;
          if ((result.score >= mateScore - searching::kMaxDepthHardCutoff && result.score <= mateScore) || !timeControlPolicy_.shouldContinue(ally)) {
            break;
          }
        }

        return result;
      };

      if (state_.getSide() == White) {
        return iterativeDeepening.template operator()<White>();
      } else {
        return iterativeDeepening.template operator()<Black>();
      }
    }
  };

  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  Board(SearchPolicy, TimePolicy) -> Board<SearchPolicy, TimePolicy>;
}
