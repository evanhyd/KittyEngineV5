#pragma once
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "boardstate.h"
#include "transposition_table.h"
#include "node_meta.h"
#include "staged_move_picker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <utility>

namespace bb::searching {
  template <evaluation::EvaluationPolicy EvalPolicy>
  class NegamaxSearchPolicy {
    // Data Structures Magic.
    static constexpr size_t kMaxKillers = 3;
    static constexpr size_t kPvSlots = (kMaxDepthHardCutoff + 1) * (kMaxDepthHardCutoff + 2) / 2;

    struct Eval {
      int32_t score;
      bool isUnreliable;
      explicit constexpr Eval(int32_t score, bool isUnreliable) noexcept
        : score(score), isUnreliable(isUnreliable) {}
    };

    struct SearchContext {
      BoardState& state;
      PositionHistory& positionHistory;
      PVLine& pvLine;
      int maxDepth; // main search max depth
      uint64_t searchedNodes = 1;
    };

    struct SearchStack {
      std::array<Move, kMaxKillers> killerMoves{};
      uint32_t pvLength = 0;
      std::optional<int32_t> staticScore;
      std::optional<bool> inCheck;

      void clearForPly() {
        pvLength = 0;
        staticScore = std::nullopt;
        inCheck = std::nullopt;
      }

      bool isKillerMove(Move move) const noexcept {
        return std::find(killerMoves.begin(), killerMoves.end(), move) != killerMoves.end();
      }
    };

    template <typename T>
    using ButterFlyTable = std::array<std::array<T, kSquareSize>, kSquareSize>;

    // Pruning Magic.
    static constexpr int kFutilityDepth = 3;
    static constexpr int32_t kFutilityMarginPerDepth = 100;
    static constexpr int kReverseFutilityDepth = 3;
    static constexpr int32_t kReverseFutilityMarginPerDepth = 100;
    static_assert(kFutilityDepth == kReverseFutilityDepth, "allow for lazy static eval optimization");

    // https://chessprogramming.org/Late_Move_Reductions#reduction-depth
    static constexpr int kLateMoveReductionDepth = 3;
    static inline const auto kLmrReductions = [] {
      std::array<std::array<uint8_t, MoveList{}.capacity()>, kMaxDepthHardCutoff+1> table{};
      for (size_t depth = kLateMoveReductionDepth; depth < table.size(); ++depth) {
        const double lnDepth = std::log(double(depth));
        for (size_t i = 1; i < table[depth].size(); ++i) {
          table[depth][i] = uint8_t(std::min(double(depth - 2 + 0.1), 0.7844 + lnDepth * std::log(double(i)) / 2.4696));
        }
      }
      return table;
    }();

    constexpr int lateMovePruningThreshold(int depth) noexcept {
      return 3 + 2 * depth * depth;
    }


    // PRIVATE MEMBERS
    EvalPolicy evalPolicy_;
    const int32_t aspirationWindow_;
    std::array<SearchStack, kMaxDepthHardCutoff + 1> searchStack_;
    std::array<Move, kPvSlots> pvTable_{}; // PV row at ply p has room for at most 65 - p moves.
    TranspositionTable ttTable_;

    // History Heuristics
    std::array<ButterFlyTable<int32_t>, 2> historyTable_{};

    int getPly(const SearchStack* frame) const noexcept {
      return static_cast<int>(frame - searchStack_.data());
    }

    static constexpr size_t pvOffset(size_t ply) noexcept {
      return ply * (2 * (kMaxDepthHardCutoff + 1) - ply + 1) / 2;
    }

    Move* pvMoves(SearchStack* frame) noexcept {
      return pvTable_.data() + pvOffset(static_cast<size_t>(getPly(frame)));
    }

    void acceptChildPVLine(SearchStack* frame, Move move) noexcept {
      const uint32_t length = (frame + 1)->pvLength + 1;
      Move* line = pvMoves(frame);
      line[0] = move;
      std::copy_n(pvMoves(frame + 1), (frame + 1)->pvLength, line + 1);
      frame->pvLength = length;
    }

    static bool isCheckmateScore(int32_t score) noexcept {
      return evaluation::kCheckmateWinScore - std::abs(score) <= kMaxDepthHardCutoff;
    }

    // Convert between root-relative and node-relative checkmate distance penalties.
    static int32_t normalizeScore(int32_t score, int ply) noexcept {
      if (!isCheckmateScore(score)) {
        return score;
      }
      return score < 0 ? score - ply : score + ply;
    }

    static int32_t denormalizeScore(int32_t score, int ply) noexcept {
      return normalizeScore(score, -ply);
    }

    static std::optional<Eval> checkDraw(const SearchContext& context, int ply) noexcept {
      const auto halfmove = context.state.getHalfmove();
      if (halfmove < 4) {
        return std::nullopt;
      }
      if (halfmove >= 100) {
        return Eval(0, true);
      }
      auto it = context.positionHistory.rbegin();
      const auto end = it + std::min(context.positionHistory.size(), static_cast<size_t>(halfmove) + 1);
      // Don't use std::find, because it uses ! instead of <, and "end" might be before "it".
      for (++it; it < end; ++it) {
        if (*it == context.state.getHash()) {
          const bool inGameHistory = int(std::distance(it, end)) >= ply;
          return Eval(0, !inGameHistory);
        }
      }
      return std::nullopt;
    }

    int32_t expandWindow(int attempt) const noexcept {
      switch (attempt) {
      case 1: return 3 * aspirationWindow_;
      case 2: return 6 * aspirationWindow_;
      case 3: return 12 * aspirationWindow_;
      default: return evaluation::kCheckmateWinScore;
      }
    };

    void storeTT(const SearchContext& context, SearchStack* frame, int depth,
                 TranspositionTable::ScoreType scoreType, Eval eval, Move bestMove, bool inCheck) noexcept {
      if (!eval.isUnreliable) {
        ttTable_.put(TranspositionTable::Entry{
          .key = context.state.getHash(),
          .score = normalizeScore(eval.score, getPly(frame)),
          .staticScore = frame->staticScore.value_or(0),
          .bestMove = bestMove,
          .depth = static_cast<int16_t>(depth),
          .age = 0,
          .scoreType = scoreType,
          .hasStaticScore = frame->staticScore.has_value(),
          .inCheck = inCheck,
          .hasInCheck = true,
        });
      }
    }

    template <NodeMeta meta>
    std::optional<Eval> probeTT(const SearchContext& context, SearchStack* frame, int depth,
                                int32_t alpha, int32_t beta, Move& ttMove) noexcept {
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return std::nullopt;
      }
      ttMove = ttEntry->bestMove;

      // Retrieve static position evaluation and inCheck regardless of TT depth.
      if (ttEntry->hasStaticScore) {
        frame->staticScore = ttEntry->staticScore;
      }
      if (ttEntry->hasInCheck) {
        frame->inCheck = ttEntry->inCheck;
      }

      // Ignore if depth was too shallow.
      if (ttEntry->depth < depth) {
        return std::nullopt;
      }

      // Do not allow TT cutoffs on PV nodes.
      // PV nodes must be searched to properly populate the triangular PV table and verify repetitions.
      if constexpr (meta.isPV) {
        return std::nullopt;
      } else {
        const int ply = getPly(frame);
        const int32_t score = denormalizeScore(ttEntry->score, ply);

        // Exact hits.
        if (ttEntry->scoreType == TranspositionTable::ScoreType::Exact) {
          return Eval(score, false);
        }

        // Lower upper bound hits.
        if ((ttEntry->scoreType == TranspositionTable::ScoreType::UpperBound && score <= alpha) ||
            (ttEntry->scoreType == TranspositionTable::ScoreType::LowerBound && score >= beta)) {
          return Eval(score, false);
        }
        return std::nullopt;
      }
    }

    void recordKiller(decltype(SearchStack::killerMoves)& killers, Move move) noexcept {
      const auto it = std::find(killers.begin(), killers.end(), move);
      const size_t slot = it == killers.end() ? killers.size() - 1 : static_cast<size_t>(it - killers.begin());
      for (size_t i = slot; i > 0; --i) {
        killers[i] = killers[i - 1];
      }
      killers[0] = move;
    }

    void recordHistory(Side side, Move move, int32_t reward) noexcept {
      constexpr int32_t kMaxHistoryPriority = 95; // Reserve up-to 5 killer moves
      constexpr int32_t kMinHistoryPriority = -kMaxHistoryPriority;
      int32_t& score = historyTable_[side][move.from()][move.to()];
      score = std::clamp(score + reward, kMinHistoryPriority, kMaxHistoryPriority);
    }

    void recordQuietCutoff(Side side, SearchStack* frame, const MoveList& moves, size_t index, int depth) noexcept {
      const Move move = moves[index];
      if (move.isViolentMove()) {
        return;
      }
      recordKiller(frame->killerMoves, move);
      const int reward = depth * depth;
      recordHistory(side, move, reward);
      for (size_t i = 0; i < index; ++i) {
        if (!moves[i].isViolentMove()) {
          recordHistory(side, moves[i], -reward);
        }
      }
    }

    

    //////////////////
    // SEARCH CHILD //
    //////////////////
    template <NodeMeta meta>
    Eval searchChild(SearchContext& context, SearchStack* childFrame, const int childDepth, const int32_t childAlpha, const int32_t childBeta) {
      childFrame->clearForPly();
      Eval result = searchInternal<meta>(context, childFrame, childDepth, -childBeta, -childAlpha);
      result.score = -result.score;
      return result;
    }

    template <NodeMeta meta>
    Eval searchInternal(SearchContext& context, SearchStack* frame, const int depth, const int32_t alpha, const int32_t beta) {
      ++context.searchedNodes;
      const int ply = getPly(frame);
      Move ttMove{};

      if constexpr (meta.isInternal()) {
        frame->clearForPly();

        if (auto drawEval = checkDraw(context, ply)) {
          return *drawEval;
        }
        if (auto ttEval = probeTT<meta>(context, frame, depth, alpha, beta, ttMove)) {
          return *ttEval;
        }
        if (depth <= 0) {
          --context.searchedNodes;
          return searchInternal<meta.toQuiescence()>(context, frame, 0, alpha, beta);
        }
      } else {
        // TOOD: move to the top if our search depth indeed reach 64 in the future.
        // Right now it is not even close to searching up to 64 depth.
        if (ply == kMaxDepthHardCutoff) {
          return Eval(evalPolicy_.evaluate(context.state), false);
        }
      }

      MoveList moves;
      Eval bestEval(alpha, false);
      const bool inCheck = frame->inCheck.value_or(context.state.template isInCheck<meta.side>());
      frame->inCheck = inCheck;
      const auto getStaticScore = [&]() noexcept {
        if (!frame->staticScore) {
          frame->staticScore = evalPolicy_.evaluate(context.state);
        }
        return *frame->staticScore;
      };

      if constexpr (meta.isInternal()) {
        // Checkmate or draw.
        context.state.template generateMoves<meta.side>(moves);
        if (moves.empty()) {
          return Eval(inCheck ? evaluation::kCheckmateLossScore + ply : evaluation::kStalemateScore, false);
        }

        if constexpr (!meta.isPV) {
          // Disable in check, or with low material (zugzwang risk).
          if (!inCheck && !isCheckmateScore(beta) && context.state.template countMMPiece<meta.side>() >= 2) {

            // Reverse Futility Pruning (Static Null Move Pruning).
            if (depth <= kReverseFutilityDepth && getStaticScore() - depth * kReverseFutilityMarginPerDepth >= beta) {
              const Eval eval(beta, false);
              storeTT(context, frame, depth, TranspositionTable::ScoreType::LowerBound, eval, ttMove, inCheck);
              return eval;
            }

            // Null Move Pruning. Disable near leaves. Disable consecutive null moves.
            if constexpr (meta.canNullMove) {
              if (depth >= 2 && getStaticScore() >= beta) {
                const int reduction = std::max(1, depth / 2);
                MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
                context.positionHistory.push(context.state.getHash());
                Eval eval = searchChild<meta.flip().disableNullMove()>(context, frame + 1, depth - reduction, beta - 1, beta);
                context.positionHistory.pop();
                context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);

                if (eval.score >= beta) {
                  if (isCheckmateScore(eval.score)) {
                    eval.score = beta;
                  }
                  storeTT(context, frame, depth, TranspositionTable::ScoreType::LowerBound, eval, ttMove, inCheck);
                  return eval;
                }
              }
            }
          }
        }
      } else {
        // Quiescence search: no stand-pat when in check.
        // Allow quiescence to ignore stalemate check, since it is too rare.
        if (inCheck) {
          context.state.template generateMoves<meta.side>(moves);
          if (moves.empty()) {
            return Eval(evaluation::kCheckmateLossScore + ply, false);
          }
        } else {
          const int32_t standPat = getStaticScore();
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

      Move bestMove{};

      // Explore moves.
      StagedMovePicker movePicker;
      for (size_t i = 0; i < moves.size(); ++i) {
        const Move move = movePicker.pick<meta>(moves, ttMove, context.state, frame->killerMoves, historyTable_);

        if constexpr (meta.isInternal() && !meta.isPV) {
          // Futility Pruning.
          if ((i > 0) && (depth <= kFutilityDepth) && !inCheck && !move.isViolentMove() && getStaticScore() + kFutilityMarginPerDepth * depth <= bestEval.score) {
            continue;
          }
        } else if constexpr (meta.isQuiescence()) {
          // Skip bad violent move if no check is involved.
          if (!inCheck && movePicker.isBadViolentMove()) {
            break;
          }
        }

        const MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);

        context.positionHistory.push(context.state.getHash());
        //ttTable_.prefetch(context.state.getHash());

        const Eval eval = [&] {
          if constexpr (meta.isInternal()) {
            constexpr auto childMeta = meta.flip().enableNullMove();

            // First move: full window. Subsequent moves: scout search.
            if (i == 0) {
              return searchChild<childMeta.setPV(meta.isPV)>(context, frame + 1, depth - 1, bestEval.score, beta);
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (!inCheck && (!move.isViolentMove() || movePicker.isBadViolentMove())) {
              reduction = int(kLmrReductions[depth][i]);
            }
            Eval result = searchChild<childMeta.setPV(false)>(context, frame + 1, depth - 1 - reduction, bestEval.score, bestEval.score + 1);

            // If LMR improves alpha, research with scout.
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, frame + 1, depth - 1, bestEval.score, bestEval.score + 1);
            }

            // If within window, full search.
            // Non PV can stop, because it only needed to prove improvement.
            if constexpr (meta.isPV) {
              if (bestEval.score < result.score && result.score < beta) {
                result = searchChild<childMeta.setPV(true)>(context, frame + 1, depth - 1, bestEval.score, beta);
              }
            }
            return result;
          } else {
            return searchChild<meta.flip().setPV(false)>(context, frame + 1, depth - 1, bestEval.score, beta);
          }
        }();

        context.positionHistory.pop();
        context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);

        if (eval.score >= beta) {
          if constexpr (meta.isInternal()) {
            storeTT(context, frame, depth, TranspositionTable::ScoreType::LowerBound, eval, move, inCheck);
            recordQuietCutoff(meta.side, frame, moves, i, depth);
          }
          return eval;
        }
        if (eval.score > bestEval.score) {
          bestEval = eval;
          if constexpr (meta.isInternal()) {
            bestMove = move;
            acceptChildPVLine(frame, bestMove);
          }
        }
      }
      
      if constexpr (meta.isInternal()) {
        const auto scoreType = (bestEval.score <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact);
        storeTT(context, frame, depth, scoreType, bestEval, bestMove ? bestMove : moves[0], inCheck);
      }
      return bestEval;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, int32_t aspirationWindow, size_t tranpositionTableSize)
      : evalPolicy_(std::move(evalPolicy)), aspirationWindow_(aspirationWindow), ttTable_(tranpositionTableSize) {
    }

    void reset() {
      evalPolicy_.reset();
      historyTable_ = {};
      ttTable_.clear();
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
      constexpr NodeMeta meta(ally, NodeMeta::NodeType::Internal, true, true);
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

      searchStack_[0].clearForPly();

      // Lookup transposition table for move ordering.
      Move ttMove{};
      if (auto ttEntry = ttTable_.get(state.getHash())) {
        ttMove = ttEntry->bestMove;
        if (ttEntry->hasStaticScore) {
          searchStack_[0].staticScore = ttEntry->staticScore;
        }
        if (ttEntry->hasInCheck) {
          searchStack_[0].inCheck = ttEntry->inCheck;
        }
      }

      // Generate root moves for the first time.
      if (!param.rootMoves) {
        param.rootMoves.emplace();
        state.generateMoves<ally>(*param.rootMoves);
        std::ranges::for_each(searchStack_, [](auto& stack) { stack.killerMoves = {}; });
        ttTable_.age();
      }
      MoveList& moves = *param.rootMoves;
      if (moves.empty()) {
        context.pvLine.clear();
        const int32_t score = state.isInCheck<ally>() ? evaluation::kCheckmateLossScore : evaluation::kStalemateScore;
        return makeResult(score, std::nullopt);
      }

      // Set up aspiration window.
      int searchAttempt = 0;
      auto [alpha, beta] = context.maxDepth <= 4
        ? std::pair{evaluation::kCheckmateLossScore, evaluation::kCheckmateWinScore}
        : std::pair{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};

      const bool inCheck = searchStack_[0].inCheck.value_or(state.isInCheck<ally>());
      searchStack_[0].inCheck = inCheck;

      // Search for the best move.
      for (;;) {
        Eval bestEval(alpha, false);
        Move bestMove{};

        StagedMovePicker movePicker;
        for (size_t i = 0; i < moves.size(); ++i) {
          const Move move = movePicker.pick<meta>(moves, ttMove, state, searchStack_[0].killerMoves, historyTable_);
          const MoveUndo undo = state.makeMove<ally>(move, evalPolicy_);
          param.positionHistory.push(state.getHash());

          const int depth = context.maxDepth;

          const Eval eval = [&] {
            constexpr auto childMeta = meta.flip().enableNullMove();

            // Disable narrow window on PV and near leaf.
            if (i == 0 || depth == 1) {
              return searchChild<childMeta.setPV(true)>(context, searchStack_.data() + 1, depth - 1, bestEval.score, beta);
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (!inCheck && (!move.isViolentMove() || movePicker.isBadViolentMove())) {
              reduction = int(kLmrReductions[depth][i]);
            }

            Eval result = searchChild<childMeta.setPV(false)>(context, searchStack_.data() + 1, depth - 1 - reduction, bestEval.score, bestEval.score + 1);

            // If LMR improves alpha, research with scout.
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, searchStack_.data() + 1, depth - 1, bestEval.score, bestEval.score + 1);
            }

            // If within window, full search.
            if (bestEval.score < result.score && result.score < beta) {
              result = searchChild<childMeta.setPV(true)>(context, searchStack_.data() + 1, depth - 1, bestEval.score, beta);
            }
            return result;
          }();

          param.positionHistory.pop();
          state.unmakeMove<ally>(move, undo, evalPolicy_);
          if (eval.score > bestEval.score) {
            bestEval = eval;
            bestMove = move;
            acceptChildPVLine(searchStack_.data(), bestMove);
            if (bestEval.score >= beta) {
              recordQuietCutoff(meta.side, searchStack_.data(), moves, i, context.maxDepth);
              ttMove = bestMove;
              break;
            }
          }
        }

        if (!isCheckmateScore(bestEval.score)) {
          ++searchAttempt;
          if (bestEval.score >= beta) {
            beta = std::min(evaluation::kCheckmateWinScore, beta + expandWindow(searchAttempt));
            continue;
          } else if (bestEval.score <= alpha) {
            alpha = std::max(evaluation::kCheckmateLossScore, alpha - expandWindow(searchAttempt));
            continue;
          }
        }

        // Update TT and PV.
        storeTT(context, searchStack_.data(), context.maxDepth, TranspositionTable::ScoreType::Exact, bestEval, bestMove ? bestMove : moves[0], inCheck);
        param.pvLine.resize(searchStack_[0].pvLength);
        std::copy_n(pvMoves(searchStack_.data()), searchStack_[0].pvLength, param.pvLine.begin());
        return makeResult(bestEval.score, bestMove);
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
