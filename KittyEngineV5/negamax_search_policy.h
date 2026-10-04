#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "searching_policy.h"
#include <algorithm>
#include <optional>
#include <ranges>
#include <utility>

namespace bb::searching {
  template <evaluation::EvaluationPolicy EvalPolicy>
  class NegamaxSearchPolicy {

    struct NodeMeta {
      enum class NodeType {
        Root,
        Internal,
        Quiescence,
      };

      const Color ally;
      const NodeType type;
      consteval NodeMeta(Color ally, NodeType type) : ally(ally), type(type) {}
      consteval NodeMeta flip() const { return NodeMeta{ getOtherColor(ally), type }; }
      consteval NodeMeta withType(NodeType newType) const { return NodeMeta{ ally, newType }; }
    };
    static constexpr int kQuiescenceExtraDepth = 16;

    EvalPolicy evalPolicy_;
    int32_t aspirationWindow_;

    void filterViolentMoves(MoveList& moves) const {
      auto it = std::remove_if(moves.begin(), moves.end(), [](const Move& move) {
        return !move.isViolentMove();
      });
      moves.resize(std::distance(moves.begin(), it));
    }

    void sortMoves(const BoardState& state, MoveList& moves) const {
      //static constexpr int32_t kFutilityMovePriority = 0;
      //static constexpr int32_t kKillerMove = 99;
      //static constexpr int32_t kTranspositionPriority = 20000;
      //static constexpr int32_t kPrincipalVariationPriority = 10000;
      static constexpr int32_t kEnPassantPriority = 150;
      static constexpr int32_t kCastlingPriority = 151;
      static constexpr int32_t kPromotionPriority = 506;
      static constexpr std::array<std::array<int32_t, kPieceSize - 1>, kPieceSize> kCapturePriorityTable =
      { {
          {105, 205, 305, 405, 505}, // pawn
          {104, 204, 304, 404, 504}, // knight
          {103, 203, 303, 403, 503}, // bishop
          {102, 202, 302, 402, 502}, // rook
          {101, 201, 301, 401, 501}, // queen
          {100, 200, 300, 400, 500}  // king
      } };

      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      static SmallVec<PriorityMove> scoredMoves;
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
          
          if (move.getPromotedPieceType() != NoPiece) {
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
    int32_t searchInternal(BoardState& state, int maxDepth, int depth, int32_t alpha, int32_t beta, uint64_t& searchedNodes) {
      ++searchedNodes;

      if constexpr (meta.type == NodeMeta::NodeType::Internal) {
        // Perform quiescence search for a few extra depth.
        if (depth == maxDepth) {
          --searchedNodes;
          return searchInternal<meta.withType(NodeMeta::NodeType::Quiescence)>(state, maxDepth + kQuiescenceExtraDepth, depth, alpha, beta, searchedNodes);
        }
      } if constexpr (meta.type == NodeMeta::NodeType::Quiescence) {
        // Hard cutoff.
        if (depth == maxDepth) {
          return evalPolicy_.evaluate(state);
        }
      }

      // Generate moves.
      MoveList moves;
      state.generateMoves<meta.ally>(moves);

      // Check for checkmate or stalemate.
      const bool inCheck = state.isInCheck<meta.ally>();
      if (moves.empty()) {
        if (inCheck) {
          return evaluation::kCheckmateScore + depth;
        } else {
          return evaluation::kStalemateScore;
        }
      }

      if constexpr (meta.type == NodeMeta::NodeType::Quiescence) {
        if (!inCheck) {
          // Can choose not to recapture if badtrade.
          int32_t standPat = evalPolicy_.evaluate(state);
          if (standPat >= beta) {
            return beta;
          }
          alpha = std::max(alpha, standPat);
          filterViolentMoves(moves);
        } else {
          // In-check extension. Continue the search.
          // Branches are limited, so shouldn't take too long.
          // TODO: implement 3-fold repetition check.
        }
      }


      // Move ordering.
      sortMoves(state, moves);

      // Explore moves.
      for (const Move& move : moves) {
        MoveUndo undo = state.makeMove<meta.ally>(move);
        int32_t score = -searchInternal<meta.flip()>(state, maxDepth, depth + 1, -beta, -alpha, searchedNodes);
        state.unmakeMove<meta.ally>(move, undo);

        if (score >= beta) {
          return beta;
        }
        alpha = std::max(alpha, score);
      }

      return alpha;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow) {
    }

    template <Color ally>
    SearchResult search(BoardState& state, const SearchParam& param) {
      // Meta data.
      static constexpr NodeMeta rootMeta(ally, NodeMeta::NodeType::Root);

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
      sortMoves(state, moves);

      // Set up aspiration window.
      static constexpr auto expandWindow = [](int& failTime, int32_t window) {
        ++failTime;
        switch (failTime) {
        case 1:
          return 3 * window;
        case 2:
          return 10 * window;
        default:
          return -evaluation::kCheckmateScore;
        }
      };
      int failLowCount = 0;
      int failHighCount = 0;
      auto [initialAlpha, initialBeta] = [&]() {
        if (param.maxDepth <= 4) {
          return std::array<int32_t, 2>{evaluation::kCheckmateScore, -evaluation::kCheckmateScore};
        } else {
          return std::array<int32_t, 2>{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};
        }
      }();

      // Search for the best move.
      for (;;) {
        int32_t alpha = initialAlpha;
        Move bestMove = moves[0];
        for (const Move& move : moves) {
          const MoveUndo undo = state.makeMove<ally>(move);
          const int32_t score = -searchInternal<rootMeta.flip().withType(NodeMeta::NodeType::Internal)>(state, param.maxDepth, 1, -initialBeta, -alpha, searchedNodes);
          state.unmakeMove<ally>(move, undo);

          if (score > alpha) {
            alpha = score;
            bestMove = move;
          }
          if (alpha >= initialBeta) {
            break;
          }
          
        }

        if (alpha >= initialBeta) {
          // Fail-high, the position is better than expected, increase beta and re-search.
          initialBeta = std::min(-evaluation::kCheckmateScore, initialBeta + expandWindow(failHighCount, aspirationWindow_));
          continue;
        } else if (alpha <= initialAlpha) {
          // Fail-low, the position is worse than expected, decrease alpha and re-search.
          initialAlpha = std::max(evaluation::kCheckmateScore, initialAlpha - expandWindow(failLowCount, aspirationWindow_));
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
