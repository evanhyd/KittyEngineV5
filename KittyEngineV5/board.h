#pragma once
#include "boardstate.h"
#include "notation.h"
#include "position_fens.h"
#include "searching_policy.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <format>
#include <optional>
#include <span>
#include <cstddef>
#include <vector>

namespace bb {
  template <typename SearchPolicy>
  class Board {
    SearchPolicy searchPolicy_;
    BoardState state_;
    int32_t historicalEval_;
    std::vector<Move> historicalMoves_;

  public:
    explicit Board(SearchPolicy searchPolicy)
      : searchPolicy_(std::move(searchPolicy)) {
      setPosition(fen::kStartPosition);
      historicalMoves_.reserve(256);
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
      historicalEval_ = 0;
      historicalMoves_.clear();
    }

    void setPosition(std::string_view fen, std::span<const std::string_view> moves) {
      const BoardState previousState = state_;
      const int32_t previousEval = historicalEval_;
      std::vector<Move> previousMoves = std::move(historicalMoves_);
      try {
        setPosition(fen);
        for (const std::string_view moveText : moves) {
          playMove(moveText);
        }
      } catch (...) {
        state_ = previousState;
        historicalEval_ = previousEval;
        historicalMoves_ = std::move(previousMoves);
        throw;
      }
    }

    searching::SearchResult search(int maxDepth, auto resultCallback) {
      const auto iterativeDeepening = [&]<Color ally>() {
        searching::SearchResult result;
        for (int depth = 1; depth <= maxDepth; ++depth) {
          const searching::SearchParam param{
            .maxDepth = depth,
            .historicalEval = historicalEval_,
            .pvMove = std::nullopt
          };
          result = searchPolicy_.template search<ally>(state_, param);
          historicalEval_ = result.score;
          resultCallback(result);
        }

        return result;
      };

      if (state_.getColorToMove() == kWhite) {
        return iterativeDeepening.template operator()<kWhite>();
      } else {
        return iterativeDeepening.template operator()<kBlack>();
      }
    }

    constexpr void playMove(std::string_view moveText) {
      const auto playMoveImpl = [&]<Color ally>() {
        MoveList moves;
        state_.generateMoves<ally>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<ally>(move);
            historicalMoves_.push_back(move);
            return;
          }
        }
        throw std::invalid_argument(std::format("Illegal UCI move: {}", moveText));
      };

      if (state_.getColorToMove() == kWhite) {
        playMoveImpl.template operator()<kWhite>();
      } else {
        playMoveImpl.template operator()<kBlack>();
      }
    }
  };

  template <typename SearchPolicy>
  Board(SearchPolicy) -> Board<SearchPolicy>;
}
