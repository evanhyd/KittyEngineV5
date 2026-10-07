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
    searching::PositionHistory positionHistory_;
    searching::MoveHistory moveHistory_;
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
      positionHistory_.clear();
      positionHistory_.push(state_.getHash());
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

        // Keep the old PV if replay ends at the same position, or advance it by one move if it ends at the PV successor.
        if (state_.getHash() == previousState.getHash()) {
          pvLine_ = previousPV;
        } else if (!previousPV.empty()) {
          const auto matchesExpectedSuccessor = [&]<Side ally>() {
            MoveList legalMoves;
            previousState.generateMoves<ally>(legalMoves);
            const Move pvMove = previousPV.front();
            if (std::find(legalMoves.begin(), legalMoves.end(), pvMove) == legalMoves.end()) {
              return false;
            }
            BoardState nextState = previousState;
            nextState.makeMove<ally>(pvMove);
            return nextState.getHash() == state_.getHash();
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
        positionHistory_ = previousPositionHistory;
        moveHistory_ = previousMoveHistory;
        pvLine_ = previousPV;
        throw;
      }
    }

    constexpr void playMove(std::string_view moveText) {
      const auto playMoveImpl = [&]<Side ally>() {
        MoveList moves;
        state_.generateMoves<ally>(moves);
        for (const Move& move : moves) {
          if (notation::moveToString(move) == moveText) {
            state_.makeMove<ally>(move);
            positionHistory_.push(state_.getHash());
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

      const auto iterativeDeepening = [&]<Side ally>() {
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
          result = searchingPolicy_.template search<ally>(state_, param);
          resultCallback(result);

          if (!timeControlPolicy_.shouldContinue(ally)) {
            break;
          }
        }

        return result;
      };

      if (state_.getSideToMove() == White) {
        return iterativeDeepening.template operator() < White > ();
      } else {
        return iterativeDeepening.template operator() < Black > ();
      }
    }
  };

  template <searching::SearchingPolicy SearchPolicy, time_control::TimeControlPolicy TimePolicy>
  Board(SearchPolicy, TimePolicy) -> Board<SearchPolicy, TimePolicy>;
}
