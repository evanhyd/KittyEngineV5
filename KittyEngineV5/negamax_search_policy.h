#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "searching_policy.h"
#include <algorithm>
#include <optional>
#include <ranges>
#include <utility>

namespace bb::searching {
  struct NodeMeta {
    Color ally;
  };

  template <evaluation::EvaluationPolicy EvalPolicy>
  class NegamaxSearchPolicy {
    EvalPolicy evalPolicy_;
    int32_t aspirationWindow_ = 50;

    //static constexpr int32_t kFutilityMovePriority = 0;
    //static constexpr int32_t kKillerMove = 99;
    //static constexpr int32_t kTranspositionPriority = 20000;
    static constexpr int32_t kEnPassantPriority = 150;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kPromotionPriority = 506;
    static constexpr int32_t kPrincipalVariationPriority = 10000;
    static constexpr std::array<std::array<int32_t, kPieceSize - 1>, kPieceSize> kCapturePriorityTable =
    {{
        {105, 205, 305, 405, 505}, // pawn
        {104, 204, 304, 404, 504}, // knight
        {103, 203, 303, 403, 503}, // bishop
        {102, 202, 302, 402, 502}, // rook
        {101, 201, 301, 401, 501}, // queen
        {100, 200, 300, 400, 500}  // king
    }};

    template <NodeMeta meta>
    void sortMoves(const BoardState& state, MoveList& moves) const {
      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      SmallVec<PriorityMove> scoredMoves;
      scoredMoves.resize(moves.size());

      // Calculate the priority.
      std::ranges::transform(moves, scoredMoves.begin(), [&](const Move& move) {
        int32_t priority = 0;
        if (move.isEnpassant()) {
          priority += kEnPassantPriority;
        } else if (move.isCastling()) {
          priority += kCastlingPriority;
        } else {
          if (move.isCapture()) {
              priority += kCapturePriorityTable[move.getMovedPiece()][std::get<1>(*state.getPieceAt(move.getDest()))];
          }
          
          if (move.getPromotedPieceType() != kNoPiece) {
            priority += kPromotionPriority;
          }
        }

        return PriorityMove{ priority, move };
      });

      // Sort the moves based on priority, then map back to the original moves list.
      std::ranges::sort(scoredMoves, std::greater<>{}, &PriorityMove::priority);
      std::ranges::transform(scoredMoves, moves.begin(), &PriorityMove::move);
    }

    template <NodeMeta meta>
    int32_t search(BoardState& state, const int maxDepth, int depth, int32_t alpha, int32_t beta, uint64_t& searchedNodes) {
      ++searchedNodes;

      // Evaluate at leaf node.
      if (depth == maxDepth) {
        return evalPolicy_.evaluate(state);
      }

      MoveList moves;
      state.generateMoves<meta.ally>(moves);

      // Check for checkmate or stalemate.
      if (moves.empty()) {
        if (state.isInCheck<meta.ally>()) {
          return evaluation::kCheckmateScore + depth;
        } else {
          return evaluation::kStalemateScore;
        }
      }

      // Move ordering.
      sortMoves<meta>(state, moves);

      // Explore moves.
      for (const Move& move : moves) {
        MoveUndo undo = state.makeMove<meta.ally>(move);
        int32_t score = -search < NodeMeta{ getOtherColor(meta.ally) } > (state, maxDepth, depth + 1, -beta, -alpha, searchedNodes);
        state.unmakeMove<meta.ally>(move, undo);

        if (score > alpha) {
          alpha = score;
          if (alpha >= beta) {
            return alpha;
          }
        }
      }

      return alpha;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow) {
    }

    template <Color ally>
    SearchResult search(BoardState& state, const SearchParam& param) {
      // Search statistics.
      const auto startTime = std::chrono::steady_clock::now();
      uint64_t searchedNodes = 1;

      // Generate legal moves and check for checkmate or stalemate.
      MoveList moves;
      state.generateMoves<ally>(moves);
      if (moves.empty()) {
        if (state.isInCheck<ally>()) {
          return SearchResult{evaluation::kCheckmateScore, std::nullopt, searchedNodes,
                              std::chrono::steady_clock::now() - startTime};
        } else {
          return SearchResult{evaluation::kStalemateScore, std::nullopt, searchedNodes,
                              std::chrono::steady_clock::now() - startTime};
        }
      }
      sortMoves<NodeMeta{ally}>(state, moves);

      // Set up aspiration window.
      int failLowCount = 0;
      int failHighCount = 0;
      int32_t initialAlpha = param.historicalEval - aspirationWindow_;
      int32_t initialBeta = param.historicalEval + aspirationWindow_;

      static constexpr auto cube = [](int32_t x) { return x * x * x; };

      // Search for the best move.
      for (;;) {
        int32_t alpha = initialAlpha;
        Move bestMove;
        for (const Move& move : moves) {
          const MoveUndo undo = state.makeMove<ally>(move);
          const int32_t score = -search < NodeMeta{ getOtherColor(ally) } > (state, param.maxDepth, 1, -initialBeta, -alpha, searchedNodes);
          state.unmakeMove<ally>(move, undo);

          if (score > alpha) {
            alpha = score;
            bestMove = move;

            // Fail-high, increase beta and re-search.
            if (alpha >= initialBeta) {
              break;
            }
          }
        }

        if (alpha >= initialBeta) {
          // Fail-high, the position is better than expected, increase beta and re-search.
          ++failHighCount;
          initialBeta = std::min(-evaluation::kCheckmateScore, initialBeta + cube(failHighCount + 1) * aspirationWindow_);
          continue;
        } else if (alpha <= initialAlpha) {
          // Fail-low, the position is worse than expected, decrease alpha and re-search.
          ++failLowCount;
          initialAlpha = std::max(evaluation::kCheckmateScore, initialAlpha - cube(failLowCount + 1) * aspirationWindow_);
          initialBeta = std::min(-evaluation::kCheckmateScore, (initialAlpha + initialBeta) / 2);
          continue;
        }

        return SearchResult{
          .score = alpha,
          .bestMove = bestMove,
          .nodesSearched = searchedNodes,
          .searchingTime = std::chrono::steady_clock::now() - startTime
        };
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
