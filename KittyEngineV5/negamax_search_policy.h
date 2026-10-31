#pragma once
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "boardstate.h"
#include "transposition_table.h"
#include "node_meta.h"
#include "staged_move_picker.h"
#include "history_heuristics.h"
#include "negamax_search_config.h"
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
    struct Eval {
      int32_t score;
      bool isUnreliable;
    };

    struct SearchContext {
      BoardState& state;
      const int maxDepth;
      PositionHistory& history;
      uint64_t nodes;
    };

    struct SearchFrame {
      Move move{};
      uint32_t pvLength = 0;
      std::optional<int32_t> staticScore;
      std::optional<bool> inCheck;

      void clear() noexcept {
        move = Move{};
        pvLength = 0;
        staticScore = std::nullopt;
        inCheck = std::nullopt;
      }
    };

    // PRIVATE MEMBERS
    static constexpr size_t kPvSlots = (kMaxDepthHardCutoff + 1) * (kMaxDepthHardCutoff + 2) / 2;

    EvalPolicy evalPolicy_;
    const NegamaxSearchConfig config_;
    const int32_t aspirationWindow_;
    std::array<int32_t, kSideSize> finalScore_{};
    std::array<SearchFrame, kMaxDepthHardCutoff + 1> frames_;
    std::array<Move, kPvSlots> pvTable_{}; // PV row at ply p has room for at most 65 - p moves.
    std::array<std::array<uint8_t, MoveList{}.capacity()>, kMaxDepthHardCutoff + 1> lateMoveReductions_{};
    TranspositionTable ttTable_;
    HistoryHeuristics historyHeuristics_;
    StagedMovePicker rootMovePicker_{ 0, {}, {}, historyHeuristics_ };

    int getPly(const SearchFrame* frame) const noexcept {
      return static_cast<int>(frame - frames_.data());
    }
    
    int32_t getStaticScore(const SearchContext& context, SearchFrame* frame) noexcept {
      if (!frame->staticScore) {
        frame->staticScore = evalPolicy_.evaluate(context.state);
      }
      return *frame->staticScore;
    }

    template <Side side>
    bool isInCheck(const SearchContext& context, SearchFrame* frame) noexcept {
      if (!frame->inCheck) {
        frame->inCheck = context.state.template isInCheck<side>();
      }
      return *frame->inCheck;
    }

    Move getPrevMove(const SearchFrame* frame, int pliesAgo) const noexcept {
      const int ply = getPly(frame);
      if (ply >= pliesAgo) {
        return (frame - pliesAgo)->move;
      }
      return Move{};
    }

    static constexpr size_t pvOffset(size_t ply) noexcept {
      return ply * (2 * (kMaxDepthHardCutoff + 1) - ply + 1) / 2;
    }

    Move* pvMoves(SearchFrame* frame) noexcept {
      return pvTable_.data() + pvOffset(static_cast<size_t>(getPly(frame)));
    }

    void acceptChildPVLine(SearchFrame* frame, Move move) noexcept {
      const uint32_t length = (frame + 1)->pvLength + 1;
      Move* line = pvMoves(frame);
      line[0] = move;
      std::copy_n(pvMoves(frame + 1), (frame + 1)->pvLength, line + 1);
      frame->pvLength = length;
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
      auto it = context.history.rbegin();
      const auto end = it + std::min(context.history.size(), static_cast<size_t>(halfmove) + 1);
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
      case 1: return 6 * aspirationWindow_;
      case 2: return 12 * aspirationWindow_;
      default: return evaluation::kCheckmateWinScore;
      }
    };

    template <bool isPV, TranspositionTable::ScoreType scoreType>
    void storeTT(const SearchContext& context, SearchFrame* frame, int depth,
                 Eval eval, Move bestMove, bool inCheck) noexcept {
      if (!eval.isUnreliable) {
        ttTable_.template put<isPV, scoreType>(TranspositionTable::Entry{
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
          .isPV = isPV,
        });
      }
    }

    template <NodeMeta meta, bool isRoot = false>
    std::optional<Eval> probeTT(const SearchContext& context, SearchFrame* frame, int depth,
                                int32_t alpha, int32_t beta, Move& ttMove) noexcept {
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return std::nullopt;
      }

      ttMove = ttEntry->bestMove;
      if (ttEntry->hasStaticScore) {
        frame->staticScore = ttEntry->staticScore;
      }
      if (ttEntry->hasInCheck) {
        frame->inCheck = ttEntry->inCheck;
      }

      if constexpr (isRoot) {
        return std::nullopt;
      }

      // Ignore lower depth.
      if (ttEntry->depth < depth) {
        return std::nullopt;
      }

      // Always full search pv nodes, don't return score for early cut off.
      // This helps populating the PV table and verify repetitions.
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

    //////////////////
    // SEARCH CHILD //
    //////////////////
    template <NodeMeta meta>
    Eval searchChild(SearchContext& context, SearchFrame* childFrame, const int childDepth, const int32_t childAlpha, const int32_t childBeta) {
      childFrame->clear();
      Eval result = searchInternal<meta>(context, childFrame, childDepth, -childBeta, -childAlpha);
      result.score = -result.score;
      return result;
    }

    template <NodeMeta meta>
    Eval searchInternal(SearchContext& context, SearchFrame* frame, const int depth, const int32_t alpha, const int32_t beta) {
      ++context.nodes;
      const int ply = getPly(frame);
      Move ttMove{};

      if constexpr (meta.isInternal()) {
        if (auto drawEval = checkDraw(context, ply)) {
          return *drawEval;
        }
        if (auto ttEval = probeTT<meta>(context, frame, depth, alpha, beta, ttMove)) {
          return *ttEval;
        }
        if (depth <= 0) {
          --context.nodes;
          return searchInternal<meta.toQuiescence()>(context, frame, 0, alpha, beta);
        }
      } else {
        // TOOD: move to the top if our search depth indeed reach 64 in the future.
        // Right now it is not even close to searching up to 64 depth.
        if (ply == kMaxDepthHardCutoff) {
          return Eval(getStaticScore(context, frame), false);
        }
      }

      const Move prev1 = getPrevMove(frame, 1);
      const Move prev2 = getPrevMove(frame, 2);
      StagedMovePicker movePicker(ply, prev1, prev2, historyHeuristics_);
      MoveList& moves = movePicker.getMoveBuffer();
      Eval bestEval(alpha, false);
      const bool inCheck = isInCheck<meta.side>(context, frame);

      if constexpr (meta.isInternal()) {
        // Checkmate or draw.
        context.state.template generateMoves<meta.side>(moves);
        if (moves.empty()) {
          return Eval(inCheck ? evaluation::kCheckmateLossScore + ply : evaluation::kStalemateScore, false);
        }

        if constexpr (!meta.isPV) {
          // Disable in check, or with low material (zugzwang risk).
          if (!inCheck && !isCheckmateScore(beta) && context.state.template countMMPiece<meta.side>() >= config_.kNullMoveMinPieces) {

            // Reverse Futility Pruning (Static Null Move Pruning).
            if (depth <= config_.reverseFutilityDepth && getStaticScore(context, frame) - depth * config_.reverseFutilityMarginPerDepth >= beta) {
              const Eval eval(beta, false);
              storeTT<false, TranspositionTable::ScoreType::LowerBound>(context, frame, depth, eval, ttMove, inCheck);
              return eval;
            }

            // Null Move Pruning. Disable near leaves. Disable consecutive null moves.
            if constexpr (meta.canNullMove) {
              if (depth >= config_.nullMoveDepth && getStaticScore(context, frame) >= beta) {
                const int reduction = std::max(1, depth / config_.kNullMoveReductionDivisor);
                MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
                context.history.push(context.state.getHash());
                Eval eval = searchChild<meta.flip().disableNullMove()>(context, frame + 1, depth - reduction, beta - 1, beta);
                context.history.pop();
                context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);

                if (eval.score >= beta) {
                  if (isCheckmateScore(eval.score)) {
                    eval.score = beta;
                  }
                  storeTT<false, TranspositionTable::ScoreType::LowerBound>(context, frame, depth, eval, ttMove, inCheck);
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
          const int32_t standPat = getStaticScore(context, frame);
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
      for (size_t i = 0; i < moves.size(); ++i) {
        const auto [move, see] = movePicker.pick<meta>(ttMove, context.state);
        frame->move = move;
        const bool isViolent = move.isViolentMove();

        if constexpr (meta.isInternal() && !meta.isPV) {
          if (!inCheck && i > 0) {
            if (!isViolent) {
              // Futility Pruning.
              if (depth <= config_.futilityDepth &&
                  !historyHeuristics_.isKillerMove(ply, move) &&
                  getStaticScore(context, frame) + config_.futilityMarginPerDepth * depth <= bestEval.score) {
                continue;
              }
            } else if (depth <= config_.seePruningDepth &&
                       move.getPromotedPieceType() == NoPiece &&
                       see < config_.seePruningMarginPerDepth * depth) {
              // SEE Pruning.
              continue;
            }
          }
        } else if constexpr (meta.isQuiescence()) {
          // Skip bad violent move if no check is involved.
          if (!inCheck && movePicker.isBadViolentMove()) {
            break;
          }
        }

        const MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);

        context.history.push(context.state.getHash());
        //ttTable_.prefetch(context.state.getHash());

        const Eval eval = [&] {
          if constexpr (meta.isInternal()) {
            constexpr auto childMeta = meta.flip().enableNullMove();
            const int extension = inCheck ? 1 : 0;
            const int childDepth = depth - 1 + extension;

            // First move: full window. Subsequent moves: scout search.
            if (i == 0) {
              return searchChild<childMeta.setPV(meta.isPV)>(context, frame + 1, childDepth, bestEval.score, beta);
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (depth >= config_.lateMoveReductionDepth && !inCheck && (movePicker.isBadViolentMove() || (!isViolent && !historyHeuristics_.isKillerMove(ply, move)))) {
              reduction = int(lateMoveReductions_[depth][i]);

              if (movePicker.isBadViolentMove()) {
                ++reduction;
              } else if (!isViolent) {
                const int32_t historyScore = historyHeuristics_.template getQuietScore<meta.side>(move, prev1, prev2);
                reduction += historyHeuristics_.getHistoryReduction(historyScore);
              }
              reduction = std::clamp(reduction, 0, childDepth - 1);
            }
            Eval result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth - reduction, bestEval.score, bestEval.score + 1);

            // If LMR improves alpha, research with scout.
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth, bestEval.score, bestEval.score + 1);
            }

            // If within window, full search.
            // Non PV can stop, because it only needed to prove improvement.
            if constexpr (meta.isPV) {
              if (bestEval.score < result.score && result.score < beta) {
                result = searchChild<childMeta.setPV(true)>(context, frame + 1, childDepth, bestEval.score, beta);
              }
            }
            return result;
          } else {
            return searchChild<meta.flip().setPV(false)>(context, frame + 1, depth - 1, bestEval.score, beta);
          }
        }();

        context.history.pop();
        context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);

        if (eval.score >= beta) {
          if constexpr (meta.isInternal()) {
            storeTT<meta.isPV, TranspositionTable::ScoreType::LowerBound>(context, frame, depth, eval, move, inCheck);
            if (!move.isViolentMove()) {
              historyHeuristics_.template recordQuietCutoff<meta.side>(ply, depth, moves, i, prev1, prev2);
            } else if (undo.capturedPiece != NoPiece) {
              historyHeuristics_.template recordCaptureCutoff<meta.side>(move.getMovedPiece(), move.to(), undo.capturedPiece, depth);
            }
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
        const Move storedMove = bestMove ? bestMove : moves[0];
        if constexpr (meta.isPV) {
          if (bestEval.score <= alpha) {
            storeTT<true, TranspositionTable::ScoreType::UpperBound>(context, frame, depth, bestEval, storedMove, inCheck);
          } else {
            storeTT<true, TranspositionTable::ScoreType::Exact>(context, frame, depth, bestEval, storedMove, inCheck);
          }
        } else {
          storeTT<false, TranspositionTable::ScoreType::UpperBound>(context, frame, depth, bestEval, storedMove, inCheck);
        }
      }
      return bestEval;
    }

  public:
    explicit NegamaxSearchPolicy(EvalPolicy evalPolicy, NegamaxSearchConfig searchConfig, int32_t aspirationWindow, size_t tranpositionTableSize)
      : evalPolicy_(std::move(evalPolicy)),
        config_(searchConfig),
        aspirationWindow_(aspirationWindow),
        ttTable_(tranpositionTableSize),
        historyHeuristics_(searchConfig.maxQuietHistory,
                 searchConfig.maxRefutationHistory,
                 searchConfig.maxFollowupHistory,
                 searchConfig.maxCaptureHistory,
                 searchConfig.lateMoveReductionHistoryK,
                 searchConfig.lateMoveReductionHistoryB) {
      for (size_t depth = static_cast<size_t>(std::max(0, config_.lateMoveReductionDepth)); depth < lateMoveReductions_.size(); ++depth) {
        const double lnDepth = std::log(double(depth));
        for (size_t i = 1; i < lateMoveReductions_[depth].size(); ++i) {
          lateMoveReductions_[depth][i] = uint8_t(std::min(double(depth - 2 + 0.1), config_.lateMoveReductionBase + lnDepth * std::log(double(i)) / config_.lateMoveReductionDivisor));
        }
      }
    }

    void reset() {
      evalPolicy_.reset();
      finalScore_.fill(0);
      ttTable_.clear();
      historyHeuristics_.clear();
      rootMovePicker_.clear();
    }

    void resetEvaluation() {
      evalPolicy_.reset();
    }

    static bool isCheckmateScore(int32_t score) noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      evalPolicy_.template markPiece<Add>(side, piece, square);
    }

    template <Side side>
    SearchResult search(BoardState& state, const SearchParam& param) {
      // Result statistics.
      SearchContext context{.state = state, .maxDepth = param.maxDepth, .history = param.history, .nodes = 1};

      // Prepare the frame.
      constexpr NodeMeta meta(side, NodeMeta::NodeType::Internal, true, true);
      SearchFrame* frame = frames_.data();
      frame->clear();
      const int ply = getPly(frame);
      Move ttMove{};
      probeTT<meta, true>(context, frame, context.maxDepth, 0, 0, ttMove);

      // Lazy initialization.
      if (context.maxDepth == 1 || rootMovePicker_.empty()) {
        rootMovePicker_.clear();
        state.template generateMoves<side>(rootMovePicker_.getMoveBuffer());
        historyHeuristics_.clearKillers();
        ttTable_.age();

        // Checkmate or draw.
        if (rootMovePicker_.empty()) {
          const bool inCheck = isInCheck<side>(context, frame);
          const int32_t score = inCheck ? evaluation::kCheckmateLossScore : evaluation::kStalemateScore;
          return SearchResult{.score = score, .move = Move{}, .pvLine = {}, .nodes = context.nodes, .searchingTime = {}};
        }
      }

      // Aspiration window.
      int searchAttempt = 0;
      auto [alpha, beta] = [&]() {
        if (context.maxDepth <= 4) {
          return std::pair{ evaluation::kCheckmateLossScore, evaluation::kCheckmateWinScore };
        }
        return std::pair{ finalScore_[side] - aspirationWindow_, finalScore_[side] + aspirationWindow_ };
      }();

      // Search for the best move.
      const bool inCheck = isInCheck<side>(context, frame);
      const Move prev1 = getPrevMove(frame, 1);
      const Move prev2 = getPrevMove(frame, 2);
      for (;;) {
        Eval bestEval(alpha, false);
        Move bestMove{};

        rootMovePicker_.rePick();
        const size_t totalMoves = rootMovePicker_.size();
        for (size_t i = 0; i < totalMoves; ++i) {
          const auto [move, see] = rootMovePicker_.pick<meta>(ttMove, state);

          const MoveUndo undo = state.template makeMove<side>(move, evalPolicy_);
          context.history.push(state.getHash());
          frame->move = move;

          const Eval eval = [&] {
            constexpr auto childMeta = meta.flip().enableNullMove();
            const int extension = inCheck ? 1 : 0;
            const int depth = context.maxDepth;
            const int childDepth = depth - 1 + extension;

            // Full search if not enough depth to perform reduced depth search or on first move.
            if (i == 0 || depth < config_.lateMoveReductionDepth) {
              return searchChild<childMeta.setPV(true)>(context, frame + 1, childDepth, bestEval.score, beta);
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (depth >= config_.lateMoveReductionDepth && !inCheck && (rootMovePicker_.isBadViolentMove() || (!move.isViolentMove() && !historyHeuristics_.isKillerMove(ply, move)))) {
              reduction = int(lateMoveReductions_[depth][i]);
              if (rootMovePicker_.isBadViolentMove()) {
                ++reduction;
              } else if (!move.isViolentMove()) {
                const int32_t historyScore = historyHeuristics_.template getQuietScore<meta.side>(move, prev1, prev2);
                reduction += historyHeuristics_.getHistoryReduction(historyScore);
              }
              reduction = std::clamp(reduction, 0, childDepth - 1);
            }

            Eval result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth - reduction, bestEval.score, bestEval.score + 1);

            // If LMR improves alpha, research with scout.
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth, bestEval.score, bestEval.score + 1);
            }

            // If within window, full search.
            if (bestEval.score < result.score && result.score < beta) {
              result = searchChild<childMeta.setPV(true)>(context, frame + 1, childDepth, bestEval.score, beta);
            }
            return result;
          }();

          context.history.pop();
          state.template unmakeMove<side>(move, undo, evalPolicy_);

          // Check cutoff.
          if (eval.score > bestEval.score) {
            bestEval = eval;
            bestMove = move;
            acceptChildPVLine(frame, bestMove);
            if (bestEval.score >= beta) {
              if (!move.isViolentMove()) {
                historyHeuristics_.template recordQuietCutoff<meta.side>(ply, context.maxDepth, rootMovePicker_.getMoveBuffer(), i, prev1, prev2);
              } else if (undo.capturedPiece != NoPiece) {
                historyHeuristics_.template recordCaptureCutoff<meta.side>(move.getMovedPiece(), move.to(), undo.capturedPiece, context.maxDepth);
              }
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
            bestMove = rootMovePicker_.getMoveBuffer()[0];
            alpha = std::max(evaluation::kCheckmateLossScore, alpha - expandWindow(searchAttempt));
            continue;
          }
        }

        // Update TT and PV.
        if (!bestMove) {
          bestMove = rootMovePicker_.getMoveBuffer()[0];
        }
        finalScore_[side] = bestEval.score;
        storeTT<true, TranspositionTable::ScoreType::Exact>(context, frame, context.maxDepth, bestEval, bestMove, inCheck);
        const std::span<const Move> pvSpan{pvMoves(frame), frame->pvLength};
        return SearchResult{.score = bestEval.score, .move = bestMove, .pvLine = pvSpan, .nodes = context.nodes, .searchingTime = {}};
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy, NegamaxSearchConfig, int32_t, size_t) -> NegamaxSearchPolicy<std::decay_t<EvalPolicy>>;
}
