#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "transposition_table.h"
#include <algorithm>
#include <array>
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
      consteval NodeMeta toInternal() const { return NodeMeta{ ally, NodeType::Internal }; }
      consteval NodeMeta toQuiescence() const { return NodeMeta{ ally, NodeType::Quiescence }; }
      consteval bool isInternal() const { return type == NodeType::Internal; }
      consteval bool isQuiescence() const { return type == NodeType::Quiescence; }
    };

    struct Eval {
      int32_t score;
      bool historyDependent;
      explicit constexpr Eval(int32_t score, bool historyDependent) noexcept
        : score(score), historyDependent(historyDependent) {}
    };

    struct SearchContext {
      BoardState& state;
      PositionHistory& positionHistory;
      PVLine& pvLine;
      int maxDepth; // main search max depth
      uint64_t searchedNodes = 1;
    };

    struct SearchStack {
      const int depth;
      bool isFollowingPV;
      PVLine pvLine;
    };

    // Magic constant.
    // static constexpr int32_t kFutilityMovePriority = 0;
    // static constexpr int32_t kKillerMove = 99;
    static constexpr int32_t kEnPassantPriority = 150;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kPromotionPriority = 506;
    static constexpr int32_t kTranspositionPriority = 10000;
    static constexpr int32_t kPrincipalVariationPriority = 20000;
    static constexpr std::array<std::array<int32_t, kPieceSize - 1>, kPieceSize> kCapturePriorityTable =
    { {
        {105, 205, 305, 405, 505}, // pawn
        {104, 204, 304, 404, 504}, // knight
        {103, 203, 303, 403, 503}, // bishop
        {102, 202, 302, 402, 502}, // rook
        {101, 201, 301, 401, 501}, // queen
        {100, 200, 300, 400, 500}  // king
    } };

    template <int... Indices>
    static constexpr auto makeSearchStack(std::integer_sequence<int, Indices...>) {
      return std::array<SearchStack, sizeof...(Indices)>{{SearchStack{.depth = Indices }...}};
    }

    EvalPolicy evalPolicy_;
    int32_t aspirationWindow_;
    TranspositionTable ttTable_;
    std::array<SearchStack, kMaxDepthHardCutoff + 1> searchStack_;

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

    void acceptChildPVLine(SearchStack* frame, Move move) {
      frame->pvLine.resize((frame + 1)->pvLine.size() + 1);
      frame->pvLine.front() = move;
      std::copy((frame + 1)->pvLine.begin(), (frame + 1)->pvLine.end(), frame->pvLine.begin() + 1);
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
      static SmallVec<PriorityMove> scoredMoves{};
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
    Eval searchInternal(SearchContext& context, SearchStack* frame, const int32_t alpha, const int32_t beta) {
      ++context.searchedNodes;
      Move pvMove{};
      Move ttMove{};

      // Clear the old PV written by a sibling node.
      // Load up PV.
      if constexpr (meta.isInternal()) {
        frame->pvLine.clear();
        if (frame->isFollowingPV && context.pvLine.size() > frame->depth) {
          pvMove = context.pvLine[frame->depth];
        }
      }

      
      {
        // Threefold repetition.
        // Search position up to the half-move.
        const auto hBegin = context.positionHistory.rbegin();
        const auto hEnd = hBegin + std::min(context.positionHistory.size(), static_cast<size_t>(context.state.getHalfmoveClock()) + 1);
        const auto it = std::find(hBegin + 1, hEnd, context.state.getHash());
        if (it != hEnd) {
          if (int dist = int(std::distance(hBegin, it)); dist >= frame->depth) {
            return Eval(0, false); // Position played in the actual board, insert to TT.
          }
          return Eval(0, true); // Position played in the search tree only, don't insert to TT.
        }
        if (context.state.getHalfmoveClock() >= 100) {
          return Eval(0, true);
        }
      }
      

      if constexpr (meta.isInternal()) {
        // Lookup transposition table.
        if (auto ttEntry = ttTable_.get(context.state.getHash()); ttEntry) {
          if (ttEntry->depth >= context.maxDepth - frame->depth) {
            ttEntry->score = denormalizeCheckmateScore(ttEntry->score, frame->depth);
            if (ttEntry->scoreType == TranspositionTable::ScoreType::Exact ||
                (ttEntry->scoreType == TranspositionTable::ScoreType::UpperBound && ttEntry->score <= alpha) ||
                (ttEntry->scoreType == TranspositionTable::ScoreType::LowerBound && ttEntry->score >= beta)) {

              // Retrieve POV line from context if TT move is the PV move.
              if (ttEntry->bestMove == pvMove) {
                frame->pvLine.resize(context.pvLine.size() - frame->depth);
                std::copy(context.pvLine.begin() + frame->depth, context.pvLine.end(), frame->pvLine.begin());
              } else {
                frame->pvLine.resize(1);
                frame->pvLine.front() = ttEntry->bestMove;
              }
              return Eval(ttEntry->score, false);
            }
          }
          ttMove = ttEntry->bestMove;
        }

        // Perform quiescence search if reach ther max depth.
        if (frame->depth == context.maxDepth) {
          --context.searchedNodes;
          return searchInternal<meta.toQuiescence()>(context, frame, alpha, beta);
        }
      }

      if (meta.isQuiescence() && frame->depth == kMaxDepthHardCutoff) {
        return Eval(evalPolicy_.evaluate(context.state), false);
      }

      // Generate moves.
      MoveList moves;
      context.state.template generateMoves<meta.ally>(moves);

      // Check for checkmate or stalemate.
      const bool inCheck = context.state.template isInCheck<meta.ally>();
      if (moves.empty()) {
        if (inCheck) {
          return Eval(evaluation::kCheckmateScore + frame->depth, false);
        } else {
          return Eval(evaluation::kStalemateScore, false);
        }
      }

      // Quiescence stand-pat and in-check extension.
      Eval bestEval(alpha, false);
      Move bestMove{};

      if constexpr (meta.isQuiescence()) {
        if (!inCheck) {
          // Can choose not to recapture if badtrade.
          int32_t standPat = evalPolicy_.evaluate(context.state);
          if (standPat >= beta) {
            return Eval(beta, false);
          }
          bestEval.score = std::max(alpha, standPat);
          filterViolentMoves(moves);
        } else {
          // In-check extension. Continue the search with limited branches.
          // TODO: implement 3-fold repetition check.
        }
      }

      // Move ordering.
      if constexpr (meta.isInternal()) {
        sortMoves<meta>(context.state, moves, ttMove, pvMove);
        bestMove = moves[0];
      } else {
        sortMoves<meta>(context.state, moves, Move{}, Move{});
      }

      // Explore moves.
      for (const Move& move : moves) {
        MoveUndo undo = context.state.template makeMove<meta.ally>(move);
        context.positionHistory.push(context.state.getHash());
        (frame + 1)->isFollowingPV = meta.isInternal() && frame->isFollowingPV && move == pvMove;
        Eval eval = searchInternal<meta.flip()>(context, frame + 1, -beta, -bestEval.score);
        eval.score = -eval.score;
        context.positionHistory.pop();
        context.state.template unmakeMove<meta.ally>(move, undo);

        if (eval.score >= beta) {
          if (meta.isInternal() && !eval.historyDependent) {
            ttTable_.put(TranspositionTable::Entry{
              .key = context.state.getHash(),
              .depth = context.maxDepth - frame->depth,
              .score = normalizeCheckmateScore(eval.score, frame->depth),
              .scoreType = TranspositionTable::ScoreType::LowerBound,
              .bestMove = move
            });
          }
          return eval;
        }
        if (eval.score > bestEval.score) {
          bestEval = eval;
          bestMove = move;
          if constexpr (meta.isInternal()) {
            acceptChildPVLine(frame, bestMove);
          }
        }
      }

      // Update TT and PV.
      if (meta.isInternal() && !bestEval.historyDependent) {
        ttTable_.put(TranspositionTable::Entry{
          .key = context.state.getHash(),
          .depth = context.maxDepth - frame->depth,
          .score = normalizeCheckmateScore(bestEval.score, frame->depth),
          .scoreType = (bestEval.score <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact),
          .bestMove = bestMove
        });
      }
      return bestEval;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow, size_t tranpositionTableSize)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow), ttTable_(tranpositionTableSize),
        searchStack_(makeSearchStack(std::make_integer_sequence<int, kMaxDepthHardCutoff + 1>{})) {
    }

    template <Side ally>
    SearchResult search(BoardState& state, const SearchParam& param) {
      static constexpr NodeMeta meta(ally, NodeMeta::NodeType::Root);

      // Result statistics.
      SearchContext context{state, param.positionHistory, param.pvLine,
                            std::clamp(param.maxDepth, 1, kMaxDepthHardCutoff)};
      const auto startTime = std::chrono::steady_clock::now();
      const auto elapsedTime = [&] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - startTime);
      };
      const auto makeResult = [&](int32_t score, std::optional<Move> bestMove) {
        return SearchResult{score, bestMove, context.searchedNodes, elapsedTime(), &param.pvLine};
      };

      // PV table.
      Move pvMove{};
      if (!context.pvLine.empty()) {
        pvMove = context.pvLine.front();
      }
      searchStack_[0].pvLine.clear();

      // Lookup transposition table.
      Move ttMove{};
      if (auto ttEntry = ttTable_.get(state.getHash())) {
        if (ttEntry->depth >= context.maxDepth && ttEntry->scoreType == TranspositionTable::ScoreType::Exact) {
          // Truncate if the root TT move diverges from the PV.
          if (ttEntry->bestMove != pvMove) {
            context.pvLine.resize(1);
            context.pvLine.front() = ttEntry->bestMove;
          }
          return makeResult(ttEntry->score, ttEntry->bestMove);
        }
        ttMove = ttEntry->bestMove;
      }

      // Generate legal moves and check for checkmate or stalemate.
      MoveList moves;
      state.generateMoves<ally>(moves);
      if (moves.empty()) {
        context.pvLine.clear();
        const int32_t score = state.isInCheck<ally>() ? evaluation::kCheckmateScore : evaluation::kStalemateScore;
        return makeResult(score, std::nullopt);
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
        if (context.maxDepth <= 4) {
          return std::array<int32_t, 2>{evaluation::kCheckmateScore, -evaluation::kCheckmateScore};
        } else {
          return std::array<int32_t, 2>{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};
        }
      }();

      // Search for the best move.
      for (;;) {
        Eval bestEval(alpha, false);
        Move bestMove = moves[0];

        for (const Move& move : moves) {
          const MoveUndo undo = state.makeMove<ally>(move);
          param.positionHistory.push(state.getHash());
          searchStack_[1].isFollowingPV = move == pvMove;
          Eval eval = searchInternal<meta.flip().toInternal()>(context, searchStack_.data() + 1, -beta, -bestEval.score);
          eval.score = -eval.score;
          param.positionHistory.pop();
          state.unmakeMove<ally>(move, undo);
          if (eval.score > bestEval.score) {
            bestEval = eval;
            bestMove = move;
            if (bestEval.score >= beta) {
              break;
            } else {
              // Update PV.
              acceptChildPVLine(searchStack_.data(), bestMove);
            }
          }
        }

        if (bestEval.score >= beta) {
          // Fail-high, the position is better than expected, increase beta and re-search.
          beta = std::min(-evaluation::kCheckmateScore, beta + expandWindow(failHighCount, aspirationWindow_));
          continue;
        } else if (bestEval.score <= alpha) {
          // Fail-low, the position is worse than expected, decrease alpha and re-search.
          alpha = std::max(evaluation::kCheckmateScore, alpha - expandWindow(failLowCount, aspirationWindow_));
          continue;
        }

        // Update TT and PV.
        if (!bestEval.historyDependent) {
          ttTable_.put(TranspositionTable::Entry{
          .key = state.getHash(),
          .depth = context.maxDepth,
          .score = bestEval.score,
          .scoreType = TranspositionTable::ScoreType::Exact,
          .bestMove = bestMove,
        });
        }
        param.pvLine = searchStack_[0].pvLine;
        return makeResult(bestEval.score, bestMove);
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
