#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "transposition_table.h"
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

      const Side ally;
      const NodeType type;
      consteval NodeMeta(Side ally, NodeType type) : ally(ally), type(type) {}
      consteval NodeMeta flip() const { return NodeMeta{ getOtherSide(ally), type }; }
      consteval NodeMeta withType(NodeType newType) const { return NodeMeta{ ally, newType }; }
      consteval bool isInternal() const { return type == NodeType::Internal; }
      consteval bool isQuiescence() const { return type == NodeType::Quiescence; }
    };

    // Magic constant.
    static constexpr int kQuiescenceExtraDepth = 16;
    //static constexpr int32_t kFutilityMovePriority = 0;
    //static constexpr int32_t kKillerMove = 99;
    static constexpr int32_t kEnPassantPriority = 150;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kPromotionPriority = 506;
    static constexpr int32_t kPrincipalVariationPriority = 10001;
    static constexpr int32_t kTranspositionPriority = 10000;
    static constexpr std::array<std::array<int32_t, kPieceSize - 1>, kPieceSize> kCapturePriorityTable =
    { {
        {105, 205, 305, 405, 505}, // pawn
        {104, 204, 304, 404, 504}, // knight
        {103, 203, 303, 403, 503}, // bishop
        {102, 202, 302, 402, 502}, // rook
        {101, 201, 301, 401, 501}, // queen
        {100, 200, 300, 400, 500}  // king
    } };

    using PVTable = SmallVec<PVLine, kMaxDepthHardCutoff + 1>;

    EvalPolicy evalPolicy_;
    int32_t aspirationWindow_;
    TranspositionTable ttTable_;
    PVTable pvTable_;

    // Convert root to mate distance penalty to current node to mate distance penalty.
    // Root -------------------------ThisNode------------------------ Checkmate
    //              |
    //              v
    //             depth
    constexpr int32_t normalizeCheckmateScore(int32_t score, int depth) const {
      int32_t rootToMateDepth = -evaluation::kCheckmateScore - std::abs(score);
      if (rootToMateDepth > kMaxDepthHardCutoff) {
        // Not checkmate.
        return score;
      }

      if (score < 0) {
        return score - depth;
      } else {
        return score + depth;
      }
    }

    // Convert current node to mate distance penalty to root to mate distance penalty.
    // Root -------------------------ThisNode------------------------ Checkmate
    //              |
    //              v
    //             depth
    constexpr int32_t denormalizeCheckmateScore(int32_t score, int depth) const {
      int32_t thisNodeToMateDepth = -evaluation::kCheckmateScore - std::abs(score);
      if (thisNodeToMateDepth > kMaxDepthHardCutoff) {
        // Not checkmate.
        return score;
      }

      if (score < 0) {
        return score + depth;
      } else {
        return score - depth;
      }
    }

    void updatePV(int depth, Move move) {
      pvTable_[depth].resize(pvTable_[depth + 1].size() + 1);
      pvTable_[depth].front() = move;
      std::copy(pvTable_[depth + 1].begin(), pvTable_[depth + 1].end(), pvTable_[depth].begin() + 1);
    }

    void filterViolentMoves(MoveList& moves) const {
      auto it = std::remove_if(moves.begin(), moves.end(), [](const Move& move) {
        return !move.isViolentMove();
      });
      moves.resize(std::distance(moves.begin(), it));
    }

    template <NodeMeta meta>
    void sortMoves(const BoardState& state, MoveList& moves, Move ttMove, Move pvMove) const {
      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      static SmallVec<PriorityMove> scoredMoves;
      scoredMoves.resize(moves.size());

      // Calculate the priority.
      std::ranges::transform(moves, scoredMoves.begin(), [&](const Move& move) {
        int32_t priority = 0;

        if constexpr (!meta.isQuiescence()) {
          if (move == ttMove) {
            priority += kTranspositionPriority;
          }
          if (move == pvMove) {
            priority += kPrincipalVariationPriority;
          }
        }

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

    /////////////////////
    // SEARCH INTERNAL //
    /////////////////////
    template <NodeMeta meta>
    int32_t searchInternal(BoardState& state, int maxDepth, int depth, const int32_t alpha, const int32_t beta,
                           bool isFollowingPV, const PVLine& previousPV, uint64_t& searchedNodes) {
      ++searchedNodes;

      Move ttMove{};

      if constexpr (meta.isInternal()) {
        // Lookup transposition table.
        if (auto ttEntry = ttTable_.get(state.getHash()); ttEntry) {
          if (ttEntry->depth >= maxDepth - depth) {
            ttEntry->score = denormalizeCheckmateScore(ttEntry->score, depth);
            if (ttEntry->scoreType == TranspositionTable::ScoreType::Exact ||
                (ttEntry->scoreType == TranspositionTable::ScoreType::UpperBound && ttEntry->score <= alpha) ||
                (ttEntry->scoreType == TranspositionTable::ScoreType::LowerBound && ttEntry->score >= beta)) {

              // TODO: retireve the PV moves after the truncation.
              pvTable_[depth].resize(1);
              pvTable_[depth].front() = ttEntry->bestMove;
              return ttEntry->score;
            }
          }
          ttMove = ttEntry->bestMove;
        }

        // Clear the old PV written by a sibling node.
        // Otherwise might incorrectly use sibling's old PV if this position can not raise alpha.
        pvTable_[depth].clear();

        // Perform quiescence search if reach ther max depth.
        if (depth == maxDepth) {
          --searchedNodes;
          return searchInternal<meta.withType(NodeMeta::NodeType::Quiescence)>(state, maxDepth + kQuiescenceExtraDepth, depth, alpha, beta, false, previousPV, searchedNodes);
        }
      } if constexpr (meta.isQuiescence()) {
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

      // Quiescence stand-pat and in-check extension.
      int32_t bestScore = alpha;
      Move bestMove{};

      if constexpr (meta.isQuiescence()) {
        if (!inCheck) {
          // Can choose not to recapture if badtrade.
          int32_t standPat = evalPolicy_.evaluate(state);
          if (standPat >= beta) {
            return beta;
          }
          bestScore = std::max(alpha, standPat);
          filterViolentMoves(moves);
        } else {
          // In-check extension. Continue the search with limited branches.
          // TODO: implement 3-fold repetition check.
        }
      }

      // Move ordering.
      Move pvMove{};
      if constexpr (meta.isInternal()) {
        if (isFollowingPV && previousPV.size() > depth) {
          pvMove = previousPV[depth];
        }
        sortMoves<meta>(state, moves, ttMove, pvMove);
        bestMove = moves[0];
      } else {
        sortMoves<meta>(state, moves, Move{}, Move{});
      }

      // Explore moves.
      for (const Move& move : moves) {
        MoveUndo undo = state.makeMove<meta.ally>(move);
        int32_t score{};
        if constexpr (meta.isInternal()) {
          score = -searchInternal<meta.flip()>(state, maxDepth, depth + 1, -beta, -bestScore, isFollowingPV && move == pvMove, previousPV, searchedNodes);
        } else {
          score = -searchInternal<meta.flip()>(state, maxDepth, depth + 1, -beta, -bestScore, false, previousPV, searchedNodes);
        }
        state.unmakeMove<meta.ally>(move, undo);

        if (score >= beta) {
          if constexpr (meta.isInternal()) {
            ttTable_.put(TranspositionTable::Entry{
              .key = state.getHash(),
              .depth = maxDepth - depth,
              .score = normalizeCheckmateScore(score, depth),
              .scoreType = TranspositionTable::ScoreType::LowerBound,
              .bestMove = move
            });
          }
          return beta;
        }
        if (score > bestScore) {
          bestScore = score;
          bestMove = move;
          if constexpr (meta.isInternal()) {
            updatePV(depth, bestMove);
          }
        }
      }

      // Update TT and PV.
      if constexpr (meta.isInternal()) {
        ttTable_.put(TranspositionTable::Entry{
          .key = state.getHash(),
          .depth = maxDepth - depth,
          .score = normalizeCheckmateScore(bestScore, depth),
          .scoreType = (bestScore <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact),
          .bestMove = bestMove
        });
      }
      return bestScore;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow, size_t tranpositionTableSize)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow), ttTable_(tranpositionTableSize), pvTable_() {
    }

    template <Side ally>
    SearchResult search(BoardState& state, const SearchParam& param) {
      // Meta data.
      static constexpr NodeMeta meta(ally, NodeMeta::NodeType::Root);

      // Search statistics.
      uint64_t searchedNodes = 1;
      const auto startTime = std::chrono::steady_clock::now();
      const auto elapsedTime = [&] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - startTime);
      };

      // PV table.
      Move pvMove{};
      if (!param.pvLine.empty()) {
        pvMove = param.pvLine.front();
      }
      pvTable_.resize(param.maxDepth + 1);
      pvTable_.front().clear();

      // Lookup transposition table.
      Move ttMove{};
      if (auto ttEntry = ttTable_.get(state.getHash())) {
        if (ttEntry->depth >= param.maxDepth && ttEntry->scoreType == TranspositionTable::ScoreType::Exact) {
          PVLine cachedPV = param.pvLine;
          if (param.pvLine.empty() || param.pvLine.front() != ttEntry->bestMove) {
            cachedPV.resize(1);
            cachedPV.front() = ttEntry->bestMove;
          }

          return SearchResult{
            .score = ttEntry->score,
            .bestMove = ttEntry->bestMove,
            .nodesSearched = searchedNodes,
            .searchingTime = elapsedTime(),
            .pvLine = cachedPV,
          };
        }
        ttMove = ttEntry->bestMove;
      }

      // Generate legal moves and check for checkmate or stalemate.
      MoveList moves;
      state.generateMoves<ally>(moves);
      if (moves.empty()) {
        if (state.isInCheck<ally>()) {
          return SearchResult{evaluation::kCheckmateScore, std::nullopt, searchedNodes, elapsedTime()};
        } else {
          return SearchResult{evaluation::kStalemateScore, std::nullopt, searchedNodes, elapsedTime()};
        }
      }
      sortMoves<meta>(state, moves, ttMove, pvMove);

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
      auto [alpha, beta] = [&]() {
        if (param.maxDepth <= 4) {
          return std::array<int32_t, 2>{evaluation::kCheckmateScore, -evaluation::kCheckmateScore};
        } else {
          return std::array<int32_t, 2>{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};
        }
      }();

      // Search for the best move.
      for (;;) {
        int32_t bestScore = alpha;
        Move bestMove = moves[0];

        for (const Move& move : moves) {
          const MoveUndo undo = state.makeMove<ally>(move);
          const int32_t score = -searchInternal<meta.flip().withType(NodeMeta::NodeType::Internal)>(state, param.maxDepth, 1,
                                                                                                     -beta, -bestScore, move == pvMove,
                                                                                                     param.pvLine, searchedNodes);
          state.unmakeMove<ally>(move, undo);

          if (score > bestScore) {
            bestScore = score;
            bestMove = move;

            if (bestScore >= beta) {
              break;
            } else {
              // Update PV.
              updatePV(0, bestMove);
            }
          }
        }

        if (bestScore >= beta) {
          // Fail-high, the position is better than expected, increase beta and re-search.
          beta = std::min(-evaluation::kCheckmateScore, beta + expandWindow(failHighCount, aspirationWindow_));
          continue;
        } else if (bestScore <= alpha) {
          // Fail-low, the position is worse than expected, decrease alpha and re-search.
          alpha = std::max(evaluation::kCheckmateScore, alpha - expandWindow(failLowCount, aspirationWindow_));
          continue;
        }

        // Update TT and PV.
        ttTable_.put(TranspositionTable::Entry{
          .key = state.getHash(),
          .depth = param.maxDepth,
          .score = bestScore,
          .scoreType = TranspositionTable::ScoreType::Exact,
          .bestMove = bestMove,
        });
        return SearchResult{
          .score = bestScore,
          .bestMove = bestMove,
          .nodesSearched = searchedNodes,
          .searchingTime = elapsedTime(),
          .pvLine = pvTable_.front(),
        };
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
