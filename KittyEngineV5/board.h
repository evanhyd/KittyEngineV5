#pragma once
#include "boardstate.h"
#include "searching_policy.h"
#include "time_control_policy.h"
#include "notation.h"
#include "position_fens.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <format>
#include <optional>
#include <span>
#include <cstddef>
#include <chrono>
#include <vector>

namespace bb {
  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  class Board {
    SearchPolicy searchingPolicy_;
    TimePolicy timeControlPolicy_;
    BoardState state_;
    std::vector<Move> pastMoves_;

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
    }

    void setPosition(std::string_view fen, std::span<const std::string_view> moves) {
      const BoardState previousState = state_;
      std::vector<Move> previousMoves = std::move(pastMoves_);
      try {
        setPosition(fen);
        for (const std::string_view moveText : moves) {
          playMove(moveText);
        }
      } catch (...) {
        state_ = previousState;
        pastMoves_ = std::move(previousMoves);
        throw;
      }
    }

    searching::SearchResult search(int maxDepth, const std::optional<time_control::TimeControl>& timeControl, auto resultCallback) {
      timeControlPolicy_.set(timeControl);

      const auto iterativeDeepening = [&]<Side ally>() {
        searching::SearchResult result{
          .score = 0
        };

        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{
            .maxDepth = depth,
            .pastEval = result.score,
          };
          result = searchingPolicy_.template search<ally>(state_, param);
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
}
