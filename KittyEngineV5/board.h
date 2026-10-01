#pragma once
#include "boardstate.h"
#include "notation.h"
#include "position_fens.h"
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <format>

namespace bb {
  template <typename SearchPolicy, typename EvalPolicy, typename UIPolicy>
  class Board {
    SearchPolicy searchEngine_;
    EvalPolicy evalEngine_;
    UIPolicy ui_;
    BoardState state_;

  public:
    explicit Board(SearchPolicy searchEngine, EvalPolicy evaluationEngine, UIPolicy userInterface)
      : searchEngine_(std::move(searchEngine)), evalEngine_(std::move(evaluationEngine)),
        ui_(std::move(userInterface)), state_(fen::kStartPosition) {
    }

    void run() {
      ui_.run(*this);
    }

    void setPosition(std::string_view fen) {
      state_.setPosition(fen);
    }

    void playMove(std::string_view moveText) {
      MoveList moves;
      if (state_.getColorToMove() == kWhite) {
        state_.generateMoves<kWhite>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<kWhite>(move);
            return;
          }
        }
      } else if (state_.getColorToMove() == kBlack) {
        state_.generateMoves<kBlack>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<kBlack>(move);
            return;
          }
        }
      }

      throw std::invalid_argument(std::format("Illegal UCI move: {}", moveText));
    }

    constexpr BoardState& getState() {
      return state_;
    }

    constexpr const BoardState& getState() const {
      return state_;
    }
  };

  template <typename SearchPolicy, typename EvalPolicy, typename UIPolicy>
  Board(SearchPolicy, EvalPolicy, UIPolicy) -> Board<SearchPolicy, EvalPolicy, UIPolicy>;
}
