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
    BoardState state_;
    SearchPolicy searchPolicy_;
    TimePolicy timePolicy_;
    searching::PositionHistory history_;

  public:
    explicit Board(SearchPolicy searchPolicy, TimePolicy timePolicy)
      : searchPolicy_(std::move(searchPolicy)), timePolicy_(std::move(timePolicy)) {
      setPosition(fen::kStartPosition);
    }

    constexpr BoardState& getState() {
      return state_;
    }

    constexpr const BoardState& getState() const {
      return state_;
    }

    // Start a completely new game.
    void newGame() {
      state_.setPosition(fen::kStartPosition);
      searchPolicy_.reset();
      history_.clear();
      history_.push(state_.getHash());
    }

    // Set the initial positions and replay moves.
    // Keep search heuristic: yes.
    // Keep evaluation cache: no.
    void setPosition(std::string_view fen, std::span<const std::string_view> moves = {}) {
      state_.setPosition(fen);
      searchPolicy_.resetEvaluation();
      history_.clear();
      history_.push(state_.getHash());
      for (const auto& moveText : moves) {
        playMove(moveText);
      }
    }

    // Play a single move and apply to the board.
    void playMove(std::string_view moveStr) {
      const auto playMoveImpl = [&]<Side side>() {
        MoveList moves;
        state_.generateMoves<side>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveStr) {
            state_.makeMove<side>(move, searchPolicy_);
            history_.push(state_.getHash());
            return;
          }
        }
        throw std::invalid_argument(std::format("Illegal UCI move: {}", moveStr));
      };

      if (state_.getSide() == White) {
        playMoveImpl.template operator()<White>();
      } else {
        playMoveImpl.template operator()<Black>();
      }
    }

    void search(int maxDepth, const std::optional<time_control::TimeControl>& timeControl, auto resultCallback) {
      timePolicy_.set(timeControl);
      maxDepth = std::clamp(maxDepth, 1, searching::kMaxDepthHardCutoff);

      const auto iterativeDeepening = [&]<Side side>() {
        searching::SearchResult result{};

        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{.maxDepth = depth, .history = history_};

          const auto startTime = std::chrono::steady_clock::now();
          result = searchPolicy_.template search<side>(state_, param);
          result.searchingTime = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - startTime);

          resultCallback(result);

          // Stop if times up or have winning mate.
          if (timePolicy_.isTimeUp(side) || (searchPolicy_.isCheckmateScore(result.score) && result.score > 0)) {
            break;
          }
        }
      };

      if (state_.getSide() == White) {
        iterativeDeepening.template operator()<White>();
      } else {
        iterativeDeepening.template operator()<Black>();
      }
    }
  };

  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  Board(SearchPolicy, TimePolicy) -> Board<SearchPolicy, TimePolicy>;
}
