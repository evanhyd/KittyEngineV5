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
        NullMove,
        Quiescence,
      };

      const Side side;
      const NodeType type;
      consteval NodeMeta(Side side, NodeType type) : side(side), type(type) {}
      consteval NodeMeta flip() const { return NodeMeta{ getOtherSide(side), type }; }
      consteval NodeMeta toInternal() const { return NodeMeta{ side, NodeType::Internal }; }
      consteval NodeMeta toNullMove() const { return NodeMeta{ side, NodeType::NullMove }; }
      consteval NodeMeta toQuiescence() const { return NodeMeta{ side, NodeType::Quiescence }; }
      consteval bool isRoot() const { return type == NodeType::Root; }
      consteval bool isInternal() const { return type == NodeType::Internal; }
      consteval bool isNullMove() const { return type == NodeType::NullMove; }
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
      std::array<Move, 3> killerMoves;
    };

    // Magic constant.
    // static constexpr int32_t kFutilityMovePriority = 0;
    static constexpr int32_t kKillerMovePriority = 99;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kPromotionPriority = 900;
    static constexpr int32_t kPrincipalVariationPriority = 10000;
    static constexpr int32_t kTranspositionPriority = 20000;

    // Assume enemy doesn't hang a piece for free, penalize high value capturing low value.
    static constexpr std::array<std::array<int32_t, kPieceSize - 1>, kPieceSize> kCapturePriorityTable =
    { {
        { 100, 300, 300, 500, 900 }, // pawn
        {  25, 225, 225, 425, 825 }, // knight
        {  25, 225, 225, 425, 825 }, // bishop
        { -25, 175, 175, 375, 775 }, // rook
        {-125,  75,  75, 275, 675 }, // queen
        { 100, 300, 300, 500, 900 }  // king
    } };

    template <int... Indices>
    static constexpr auto makeSearchStack(std::integer_sequence<int, Indices...>) {
      return std::array<SearchStack, sizeof...(Indices)>{{SearchStack{.depth = Indices }...}};
    }

    EvalPolicy evalPolicy_;
    const int32_t aspirationWindow_;
    TranspositionTable ttTable_;
    std::array<SearchStack, kMaxDepthHardCutoff + 1> searchStack_;

    constexpr bool isCheckmateScore(int32_t score) const {
      return -evaluation::kCheckmateScore - std::abs(score) <= kMaxDepthHardCutoff;
    }

    // Convert between root-relative and node-relative checkmate distance penalties.
    constexpr int32_t normalizeCheckmateScore(int32_t score, int depth) const {
      if (!isCheckmateScore(score)) {
        return score;
      }
      return score < 0 ? score - depth : score + depth;
    }

    constexpr int32_t denormalizeCheckmateScore(int32_t score, int depth) const {
      return normalizeCheckmateScore(score, -depth);
    }

    template <NodeMeta meta>
    void acceptChildPVLine(SearchStack* frame, Move move) {
      if constexpr (meta.isRoot() || meta.isInternal()) {
        frame->pvLine.resize((frame + 1)->pvLine.size() + 1);
        frame->pvLine.front() = move;
        std::copy((frame + 1)->pvLine.begin(), (frame + 1)->pvLine.end(), frame->pvLine.begin() + 1);
      }
    }

    template <NodeMeta meta>
    void storeTT(const SearchContext& context, int depth, Eval eval,
                 TranspositionTable::ScoreType scoreType, Move bestMove) {
      if constexpr (meta.isRoot() || meta.isInternal() || meta.isNullMove()) {
        if (!eval.historyDependent) {
          ttTable_.put(TranspositionTable::Entry{
            .key = context.state.getHash(),
            .depth = context.maxDepth - depth,
            .score = normalizeCheckmateScore(eval.score, depth),
            .scoreType = scoreType,
            .bestMove = bestMove,
          });
        }
      }
    }

    std::optional<Eval> checkDraw(const SearchContext& context, int depth) const {
      const auto hBegin = context.positionHistory.rbegin();
      const auto hEnd = hBegin + std::min(context.positionHistory.size(),
                                          static_cast<size_t>(context.state.getHalfmoveClock()) + 1);
      if (const auto it = std::find(hBegin + 1, hEnd, context.state.getHash()); it != hEnd) {
        const bool inGameHistory = int(std::distance(hBegin, it)) >= depth;
        return Eval(0, !inGameHistory);
      }
      if (context.state.getHalfmoveClock() >= 100) {
        return Eval(0, false);
      }
      return std::nullopt;
    }

    template <NodeMeta meta>
    std::optional<Eval> probeTT(const SearchContext& context, SearchStack* frame,
                                int32_t alpha, int32_t beta, Move pvMove, Move& ttMove) {
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return std::nullopt;
      }
      ttMove = ttEntry->bestMove;
      if (ttEntry->depth < context.maxDepth - frame->depth) {
        return std::nullopt;
      }

      const int32_t score = denormalizeCheckmateScore(ttEntry->score, frame->depth);
      if (ttEntry->scoreType == TranspositionTable::ScoreType::Exact) {
        if constexpr (meta.isInternal()) {
          if (ttEntry->bestMove == pvMove) {
            frame->pvLine.resize(context.pvLine.size() - frame->depth);
            std::copy(context.pvLine.begin() + frame->depth, context.pvLine.end(), frame->pvLine.begin());
          } else {
            frame->pvLine.resize(1);
            frame->pvLine.front() = ttEntry->bestMove;
          }
        }
        return Eval(score, false);
      }
      if ((ttEntry->scoreType == TranspositionTable::ScoreType::UpperBound && score <= alpha) ||
          (ttEntry->scoreType == TranspositionTable::ScoreType::LowerBound && score >= beta)) {
        return Eval(score, false);
      }
      return std::nullopt;
    }

    void recordKiller(decltype(SearchStack::killerMoves)& killers, Move move) {
      const auto it = std::find(killers.begin(), killers.end(), move);
      const size_t slot = it == killers.end() ? killers.size() - 1 : static_cast<size_t>(it - killers.begin());
      for (size_t i = slot; i > 0; --i) {
        killers[i] = killers[i - 1];
      }
      killers[0] = move;
    }

    template <NodeMeta meta>
    void sortMoves(BoardState& state, MoveList& moves, Move ttMove, Move pvMove, const decltype(SearchStack::killerMoves)& killerMoves) const {
      struct PriorityMove {
        int32_t priority;
        Move move;
      };
      static SmallVec<PriorityMove> scoredMoves{};
      scoredMoves.resize(moves.size());

      // Calculate the priority.
      std::ranges::transform(moves, scoredMoves.begin(), [&](const Move& move) {
        int32_t priority = 0;

        if constexpr (meta.isRoot() || meta.isInternal() || meta.isNullMove()) {
          if (move == ttMove) {
            priority += kTranspositionPriority;
          }
          if (move == pvMove) {
            priority += kPrincipalVariationPriority;
          }
          if constexpr (!meta.isNullMove()) {
            // Disblaek killer move on null move search, because it skips depth.
            for (size_t i = 0; i < killerMoves.size() && killerMoves[i] != Move{}; ++i) {
              if (move == killerMoves[i]) {
                priority += kKillerMovePriority - int32_t(i);
              }
            }
          }
        }

        if (move.isEnpassant()) {
          priority += kCapturePriorityTable[Pawn][Pawn];
        } if (!meta.isQuiescence() && move.isCastling()) {
          priority += kCastlingPriority;
        } else {
          if (move.isCapture()) {
            Piece capturedPiece = std::get<1>(*state.getPieceAt(move.getDest()));
            priority += kCapturePriorityTable[move.getMovedPiece()][capturedPiece];
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

      // Clear the old PV written by a sibling node and load the expected PV move.
      if constexpr (meta.isInternal()) {
        frame->pvLine.clear();
        if (frame->isFollowingPV && context.pvLine.size() > frame->depth) {
          pvMove = context.pvLine[frame->depth];
        }
      }

      if (auto drawEval = checkDraw(context, frame->depth)) {
        return *drawEval;
      }

      if constexpr (!meta.isQuiescence()) {
        if (auto ttEval = probeTT<meta>(context, frame, alpha, beta, pvMove, ttMove); ttEval) {
          return *ttEval;
        }
        if (frame->depth == context.maxDepth) {
          --context.searchedNodes;
          return searchInternal<meta.toQuiescence()>(context, frame, alpha, beta);
        }
      } else if (frame->depth == kMaxDepthHardCutoff) {
        return Eval(evalPolicy_.evaluate(context.state), false);
      }

      MoveList moves;
      Eval bestEval(alpha, false);

      const bool inCheck = context.state.template isInCheck<meta.side>();
      if constexpr (meta.isInternal() || meta.isNullMove()) {
        context.state.template generateMoves<meta.side>(moves);
        if (moves.empty()) {
          return Eval(inCheck ? evaluation::kCheckmateScore + frame->depth : evaluation::kStalemateScore, false);
        }

        // Null Move Pruning: disabled near leaves, on PV, in check, or with low material (zugzwang risk).
        if (const int remainDepth = context.maxDepth - frame->depth;
            !meta.isNullMove() && remainDepth > 1 && !frame->isFollowingPV && !inCheck &&
            context.state.template countMMPiece<meta.side>() >= 2) {
          const int skipped = std::max(1, remainDepth / 2);
          MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
          context.positionHistory.push(context.state.getHash());
          Eval eval = searchInternal<meta.flip().toNullMove()>(context, frame + skipped, -beta, -beta + 1);
          context.positionHistory.pop();
          context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);
          eval.score = -eval.score;
          if (eval.score >= beta) {
            if (isCheckmateScore(eval.score)) {
              eval.score = beta;
            }
            storeTT<meta>(context, frame->depth, eval, TranspositionTable::ScoreType::LowerBound, ttMove);
            return eval;
          }
        }
      } else if (meta.isQuiescence()) {
        // Quiescence search: no stand-pat when in check.
        // Allow quiescence to ignore stalemate check, since it is too rare.
        if (inCheck) {
          context.state.template generateMoves<meta.side>(moves);
          if (moves.empty()) {
            return Eval(evaluation::kCheckmateScore + frame->depth, false);
          }
        } else {
          const int32_t standPat = evalPolicy_.evaluate(context.state);
          if (standPat >= beta) {
            return Eval(beta, false);
          }
          bestEval.score = std::max(alpha, standPat);
          context.state.template generateMoves<meta.side, true>(moves);
          if (moves.empty()) {
            return bestEval;
          }
        }
      }

      sortMoves<meta>(context.state, moves, ttMove, pvMove, frame->killerMoves);
      Move bestMove = moves[0];

      // Explore moves.
      for (size_t i = 0; i < moves.size(); ++i) {
        Move move = moves[i];
        MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);
        context.positionHistory.push(context.state.getHash());
        (frame + 1)->isFollowingPV = meta.isInternal() && frame->isFollowingPV && move == pvMove;

        Eval eval = [&]() -> Eval {
          //Eval result = searchInternal<meta.flip()>(context, frame + 1, -beta, -bestEval.score);
          //result.score = -result.score;
          //return result;

          // Search the first move and nodes without an established best move in full.
          // Quiescence and null-move nodes retain their normal alpha-beta search.
          if (!meta.isInternal() || i == 0 || bestEval.score <= alpha) {
            Eval result = searchInternal<meta.flip()>(context, frame + 1, -beta, -bestEval.score);
            result.score = -result.score;
            return result;
          }

          // A scout needs only to determine whether this move beats the current best.
          // Keep the child an internal node so TT bounds and depth handling stay intact.
          Eval result = searchInternal<meta.flip()>(context, frame + 1, -bestEval.score - 1, -bestEval.score);
          result.score = -result.score;
          if (bestEval.score < result.score && result.score < beta) {
            result = searchInternal<meta.flip()>(context, frame + 1, -beta, -bestEval.score);
            result.score = -result.score;
          }
          return result;
        }();

        context.positionHistory.pop();
        context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);

        if (eval.score >= beta) {
          storeTT<meta>(context, frame->depth, eval, TranspositionTable::ScoreType::LowerBound, move);
          if (meta.isInternal() && !move.isViolentMove()) {
            recordKiller(frame->killerMoves, move);
          }
          return eval;
        }
        if (eval.score > bestEval.score) {
          bestEval = eval;
          bestMove = move;
          acceptChildPVLine<meta>(frame, bestMove);
        }
      }
      
      storeTT<meta>(context, 
                    frame->depth, 
                    bestEval, 
                    (bestEval.score <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact), 
                    bestMove);
      return bestEval;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow, size_t tranpositionTableSize)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow), ttTable_(tranpositionTableSize),
        searchStack_(makeSearchStack(std::make_integer_sequence<int, kMaxDepthHardCutoff + 1>{})) {
    }

    void reset() {
      evalPolicy_.reset();
      ttTable_.clear();
      for (SearchStack& frame : searchStack_) {
        frame.isFollowingPV = false;
        frame.pvLine.clear();
      }
    }

    void invalidateEvaluation() {
      evalPolicy_.reset();
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      evalPolicy_.template markPiece<Add>(side, piece, square);
    }

    template <Side ally>
    SearchResult search(BoardState& state, const SearchParam& param) {
      constexpr NodeMeta meta(ally, NodeMeta::NodeType::Root);
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

      // Generate root moves for the first time.
      if (!param.rootMoves) {
        param.rootMoves.emplace();
        state.generateMoves<ally>(*param.rootMoves);
        std::ranges::for_each(searchStack_, [](auto& stack) { stack.killerMoves = {}; });
      }
      MoveList& moves = *param.rootMoves;
      if (moves.empty()) {
        context.pvLine.clear();
        const int32_t score = state.isInCheck<ally>() ? evaluation::kCheckmateScore : evaluation::kStalemateScore;
        return makeResult(score, std::nullopt);
      }
      sortMoves<meta>(state, moves, ttMove, pvMove, searchStack_[0].killerMoves);

      // Set up aspiration window.
      constexpr auto expandWindow = [](int& failTime, int32_t window) {
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
      auto [alpha, beta] = context.maxDepth <= 4
        ? std::pair{evaluation::kCheckmateScore, -evaluation::kCheckmateScore}
        : std::pair{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};

      // Search for the best move.
      for (;;) {
        Eval bestEval(alpha, false);
        Move bestMove = moves[0];

        for (Move& move : moves) {
          const MoveUndo undo = state.makeMove<ally>(move, evalPolicy_);
          param.positionHistory.push(state.getHash());
          searchStack_[1].isFollowingPV = move == pvMove;
          Eval eval = searchInternal<meta.flip().toInternal()>(context, searchStack_.data() + 1, -beta, -bestEval.score);
          eval.score = -eval.score;
          param.positionHistory.pop();
          state.unmakeMove<ally>(move, undo, evalPolicy_);
          if (eval.score > bestEval.score) {
            bestEval = eval;
            bestMove = move;
            if (bestEval.score >= beta) {
              if (!move.isViolentMove()) {
                recordKiller(searchStack_[0].killerMoves, move);
              }
              std::swap(moves[0], move);
              break;
            }
            acceptChildPVLine<meta>(searchStack_.data(), bestMove);
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
        storeTT<meta>(context, 0, bestEval, TranspositionTable::ScoreType::Exact, bestMove);
        param.pvLine = searchStack_[0].pvLine;
        return makeResult(bestEval.score, bestMove);
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
