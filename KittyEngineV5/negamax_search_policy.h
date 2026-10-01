#pragma once
#include "boardstate.h"
#include <algorithm>
#include <ranges>

namespace bb::searching {

  struct NodeMeta {
    Color ally;
  };

  class NegamaxSearchPolicy {
    static constexpr int32_t kMateScore = -1'000'000'000;
    static constexpr int32_t kStalemateScore = 0;

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

  public:
    template <NodeMeta meta, typename EvalPolicy>
    int32_t search(EvalPolicy& evalPolicy, BoardState& state, int depth, const int maxDepth, int32_t alpha, int32_t beta) {
      // Evaluate at leaf node.
      if (depth == maxDepth) {
        return evalPolicy.evaluate(state);
      }

      MoveList moves;
      state.generateMoves<meta.ally>(moves);

      // Check for checkmate or stalemate.
      if (moves.empty()) {
        if (state.isInCheck<meta.ally>()) {
          return kMateScore + depth;
        } else {
          return kStalemateScore;
        }
      }

      // Move ordering.
      sortMoves<meta>(state, moves);

      // Explore moves.
      for (const Move& move : moves) {
        MoveUndo undo = state.makeMove<meta.ally>(move);
        int32_t score = -search<NodeMeta{getOtherColor(meta.ally)}, EvalPolicy>(evalPolicy, state, depth + 1, maxDepth, -beta, -alpha);
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
  };
}
