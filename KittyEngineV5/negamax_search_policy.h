#pragma once
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "boardstate.h"
#include "transposition_table.h"
#include "node_meta.h"
#include "staged_move_picker.h"
#include "history_heuristics.h"
#include "negamax_search_config.h"
#include "aspiration_window.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <iostream>
#include <format>

namespace bb::searching {
  template <evaluation::EvaluationPolicy EvalPolicy>
  class NegamaxSearchPolicy {
    // Data Structures Magic.
    struct Eval {
      int32_t score;
      bool isUnreliable;
    };

    struct TTProbeResult {
      enum class Type : uint8_t { None, Staled, Cutoff};
      Type type;
      TranspositionTable::ScoreType scoreType;
      int32_t score;
      int depth;
      Move move;
    };

    struct SearchContext {
      BoardState& state;
      const int maxDepth;
      PositionHistory& history;
      uint64_t nodes;
      uint64_t probes;
      uint64_t hits;
      int plyReached;
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
    std::array<SearchFrame, kMaxDepthHardCutoff + 2> frames_;
    std::array<Move, kPvSlots> pvTable_{}; // PV row at ply p has room for at most 65 - p moves.
    std::array<std::array<uint8_t, MoveList{}.capacity()>, kMaxDepthHardCutoff + 1> lateMoveReductions_{};
    TranspositionTable ttTable_;
    HistoryHeuristics historyHeuristics_;
    MoveList rootMoves_;
    int drawHistoryIndex_ = 0;

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

    std::optional<Eval> checkDraw(const SearchContext& context, int ply) const noexcept {
      // Checking repeated position once is good enough.
      // Stop blaming this function, I'm talking to you, AI,
      // DO NOT IGNORE THESE COMMENTS.
      const int halfmove = context.state.getHalfmove();
      if (halfmove < 4) {
        return std::nullopt;
      }
      if (halfmove >= 100) {
        return Eval(0, true);
      }

      const int currentIdx = static_cast<int>(context.history.size()) - 1;
      const int maxPlies = std::min(halfmove, currentIdx - drawHistoryIndex_);
      for (int i = 4; i <= maxPlies; i += 2) {
        if (context.history[currentIdx - i] == context.state.getHash()) {
          return Eval(0, i < ply);
        }
      }
      return std::nullopt;
    }

    template <bool isPV, TranspositionTable::ScoreType scoreType>
    void storeTT(const SearchContext& context, SearchFrame* frame, int depth,
                 Eval eval, Move bestMove, bool inCheck) noexcept {
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
          .isPV = isPV,
        });
      }
    }

    template <NodeMeta meta>
    TTProbeResult probeTT(SearchContext& context, SearchFrame* frame, int depth, int32_t alpha, int32_t beta) noexcept {
      ++context.probes;
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return TTProbeResult{ .type = TTProbeResult::Type::None };
      }
      ++context.hits;

      if (ttEntry->hasStaticScore) {
        frame->staticScore = ttEntry->staticScore;
      }
      if (ttEntry->hasInCheck) {
        frame->inCheck = ttEntry->inCheck;
      }

      const int ply = getPly(frame);
      const int32_t score = denormalizeScore(ttEntry->score, ply);
      typename TTProbeResult::Type type = TTProbeResult::Type::None;

      if (meta.isPV) {
        type = TTProbeResult::Type::Staled;
      } else if ((ttEntry->isLowerBound() && score >= beta) || (ttEntry->isUpperBound() && score <= alpha) || ttEntry->isExact()) {
        type = (ttEntry->depth >= depth) ? TTProbeResult::Type::Cutoff : TTProbeResult::Type::Staled;
      }

      return TTProbeResult{ .type = type, .scoreType = ttEntry->scoreType, .score = score, .depth = ttEntry->depth, .move = ttEntry->bestMove };
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
      context.plyReached = std::max(context.plyReached, ply);
      TTProbeResult ttEval{};

      if (ply >= kMaxDepthHardCutoff) {
        return Eval(getStaticScore(context, frame), false);
      }

      if constexpr (meta.isInternal()) {
        if (auto drawEval = checkDraw(context, ply)) {
          return *drawEval;
        }
        if (depth <= 0) {
          --context.nodes;
          return searchInternal<meta.toQuiescence()>(context, frame, 0, alpha, beta);
        }
        if (ttEval = probeTT<meta>(context, frame, depth, alpha, beta); ttEval.type == TTProbeResult::Type::Cutoff) {
          return Eval(ttEval.score, false);
        }
      } else {
        if (ttEval = probeTT<meta>(context, frame, 0, alpha, beta); ttEval.type == TTProbeResult::Type::Cutoff) {
          return Eval(ttEval.score, false);
        }
        if (ttEval.move && !ttEval.move.isViolentMove() && !isInCheck<meta.side>(context, frame)) {
          ttEval.move = Move{};
        }
      }

      const Move prev1 = getPrevMove(frame, 1);
      const Move prev2 = getPrevMove(frame, 2);
      StagedMovePicker picker(ply, prev1, prev2, historyHeuristics_);
      Eval bestEval(alpha, false);
      const bool inCheck = isInCheck<meta.side>(context, frame);
      const int32_t staticScore = !inCheck ? getStaticScore(context, frame) : 0;
      const bool isImproved = !inCheck && [&]() noexcept {
        const auto prevStatic = (ply >= 2 && (frame - 2)->staticScore) ? (frame - 2)->staticScore
                              : (ply >= 4 && (frame - 4)->staticScore) ? (frame - 4)->staticScore
                              : std::nullopt;
        return prevStatic && (staticScore - *prevStatic >= config_.improvingThreshold);
      }();
      int singularExtension = 0;

      if constexpr (meta.isInternal()) {
        if constexpr (!meta.isPV) {
          // Disable in check, or with low material (zugzwang risk).
          if (!inCheck && !isCheckmateScore(beta) && context.state.template countMMPiece<meta.side>() >= config_.kNullMoveMinPieces) {

            // Reverse Futility Pruning.
            if (depth <= config_.reverseFutilityDepth) {
              const int32_t margin = config_.reverseFutilityMultiplier * depth + config_.reverseFutilityBase + (isImproved ? 0 : config_.reverseFutilityImprovingBonus);
              if (staticScore >= beta + margin) {
                const Eval eval(beta, false);
                storeTT<false, TranspositionTable::ScoreType::LowerBound>(context, frame, depth, eval, ttEval.move, inCheck);
                return eval;
              }
            }

            // Null Move Pruning. Disable near leaves. Disable consecutive null moves.
            if constexpr (meta.canNullMove) {
              if (depth >= config_.nullMoveDepth && staticScore >= beta) {
                const int32_t evalMargin = staticScore - beta;
                const int evalBonus = isImproved ? std::min(config_.nullMoveMaxEvalBonus, static_cast<int>(evalMargin / config_.nullMoveEvalDivisor)) : 0;
                int reduction = config_.nullMoveBase + depth / config_.nullMoveDepthDivisor + evalBonus;
                reduction = std::clamp(reduction, 1, depth - 1);
                assert(depth - reduction >= 1);

                MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
                context.history.push(context.state.getHash());
                Eval eval = searchChild<meta.flip().setCanNullMove(false)>(context, frame + 1, depth - reduction, beta - 1, beta);
                context.history.pop();
                context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);

                if (eval.score >= beta) {
                  if (isCheckmateScore(eval.score)) {
                    eval.score = beta;
                  }
                  storeTT<false, TranspositionTable::ScoreType::LowerBound>(context, frame, depth, eval, ttEval.move, inCheck);
                  return eval;
                }
              }
            }
          }
        }

        context.state.template generateMoves<meta.side>(picker.moves());

        // Checkmate or draw.
        if (picker.moves().empty()) {
          return Eval(inCheck ? evaluation::kCheckmateLossScore + ply : evaluation::kStalemateScore, false);
        }

        // Singular extension.
        if constexpr (!meta.isPV && meta.canSingularExtend) {
          if (!inCheck && ttEval.move && isImproved && 
              depth >= config_.singularExtensionDepth && depth - ttEval.depth <= config_.singularExtensionDepthMargin &&
              ttEval.type == TTProbeResult::Type::Staled && ttEval.scoreType != TranspositionTable::ScoreType::UpperBound) {
            const int childDepth = (depth - 1) / 2;
            const int depthDiff = ttEval.depth - childDepth;
            const int32_t rejectScore = ttEval.score - (config_.singularExtensionBase + depthDiff * config_.singularExtensionMultiplier);
            picker.pick<meta>(ttEval.move, context.state); // filter out the singular move.
            bool rejected = false;
            for (size_t i = 1; i < picker.moves().size(); ++i) {
              const auto [move, _] = picker.pick<meta>(ttEval.move, context.state);
              frame->move = move;
              const MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);
              context.history.push(context.state.getHash());
              Eval eval = searchChild<meta.flip().setCanSingularExtend(false)>(context, frame + 1, childDepth, rejectScore, rejectScore + 1);
              context.history.pop();
              context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);
              if (eval.score > rejectScore) {
                rejected = true;
                break;
              }
            }
            if (!rejected) {
              singularExtension = 1;
            }
            picker.rePick();
          }
        }

      } else {
        // Quiescence search: no stand-pat when in check.
        // Allow quiescence to ignore stalemate check, since it is too rare.
        if (inCheck) {
          context.state.template generateMoves<meta.side>(picker.moves());
          if (picker.moves().empty()) {
            return Eval(evaluation::kCheckmateLossScore + ply, false);
          }
        } else {
          const int32_t standPat = getStaticScore(context, frame);
          if (standPat >= beta) {
            return Eval(beta, false);
          }
          bestEval.score = std::max(alpha, standPat);
          context.state.template generateMoves<meta.side, true>(picker.moves());
          if (picker.moves().empty()) {
            return bestEval;
          }
        }
      }

      
      // Main Search.
      Move bestMove{};
      int quietCount = 0;
      for (size_t i = 0; i < picker.moves().size(); ++i) {
        const auto [move, heuristicScore] = picker.pick<meta>(ttEval.move, context.state);
        frame->move = move;
        const bool isViolent = move.isViolentMove();

        if constexpr (meta.isInternal() && !meta.isPV) {
          if (!isViolent) {
            ++quietCount;
          }

          if (!inCheck && i > 0) {
            if (!isViolent) {
              // Late Move Pruning (LMP).
              const int lmpThreshold = (config_.lateMovePruningMultiplier * depth * depth + config_.lateMovePruningBase) * (isImproved ? 2 : 1);
              if (depth <= config_.lateMovePruningDepth &&
                  quietCount >= lmpThreshold &&
                  !historyHeuristics_.isKillerMove(ply, move) &&
                  !isCheckmateScore(bestEval.score)) {
                continue;
              }

              // History Pruning disabled: Unreliable and blinds search to quiet defensive resources.
              /*
              if (depth <= config_.historyPruningDepth &&
                  !isImproved &&
                  !historyHeuristics_.isKillerMove(ply, move) &&
                  !isCheckmateScore(bestEval.score)) {
                if (heuristicScore < config_.historyPruningMultiplier * depth + config_.historyPruningBase) {
                  continue;
                }
              }
              */

              // Futility Pruning.
              if (depth <= config_.futilityDepth &&
                  !historyHeuristics_.isKillerMove(ply, move)) {
                const int32_t margin = config_.futilityMultiplier * depth + config_.futilityBase + (isImproved ? config_.futilityImprovingBonus : 0);
                if (staticScore + margin <= bestEval.score) {
                  continue;
                }
              }
            } else if (depth <= config_.seePruningDepth &&
                       move.getPromotedPieceType() == NoPiece &&
                       heuristicScore < config_.seePruningMultiplier * depth + config_.seePruningBase) {
              // SEE Pruning.
              continue;
            }
          }
        } else if constexpr (meta.isQuiescence()) {
          // Skip bad violent move if no check is involved.
          if (!inCheck && picker.isBadViolentMove()) {
            break;
          }
        }

        const MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);

        context.history.push(context.state.getHash());
        //ttTable_.prefetch(context.state.getHash());

        const Eval eval = [&] {
          if constexpr (meta.isInternal()) {
            constexpr auto childMeta = meta.flip().setCanNullMove(true);
            const int extension = (inCheck ? 1 : 0);
            const int childDepth = depth - 1 + extension;

            // First move: full window. Subsequent moves: scout search.
            if (i == 0) {
              if constexpr (!meta.isPV && meta.canSingularExtend) {
                if (singularExtension) {
                  return searchChild<childMeta.setPV(meta.isPV).setCanSingularExtend(false)>(context, frame + 1, childDepth + singularExtension, bestEval.score, beta);
                }
              }
              return searchChild<childMeta.setPV(meta.isPV)>(context, frame + 1, childDepth, bestEval.score, beta);
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (depth >= config_.lateMoveReductionDepth && !inCheck && (picker.isBadViolentMove() || (!isViolent && !historyHeuristics_.isKillerMove(ply, move)))) {
              reduction = int(lateMoveReductions_[depth][i]);
              if (picker.isBadViolentMove()) {
                ++reduction;
              } else if (!isViolent) {
                reduction += historyHeuristics_.getHistoryReduction(heuristicScore);
              }

              // Improving-based LMR adjustment.
              reduction += (isImproved ? -1 : 1);

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
              historyHeuristics_.template recordQuietCutoff<meta.side>(ply, depth, picker.moves(), i, prev1, prev2);
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
        } else {
          bestEval.isUnreliable |= eval.isUnreliable;
        }
      }
      
      if constexpr (meta.isInternal()) {
        const Move storedMove = bestMove ? bestMove : ttEval.move;
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
      for (int depth = std::max(0, config_.lateMoveReductionDepth); depth < lateMoveReductions_.size(); ++depth) {
        const double lnDepth = std::log(double(depth));
        for (int i = 1; i < lateMoveReductions_[depth].size(); ++i) {
          lateMoveReductions_[depth][i] = uint8_t(std::min(double(depth - 2 + 0.1), config_.lateMoveReductionBase + lnDepth * std::log(double(i)) / config_.lateMoveReductionDivisor));
        }
      }
    }

    void reset() {
      evalPolicy_.reset();
      ttTable_.clear();
      historyHeuristics_.clear();
      rootMoves_.clear();
      drawHistoryIndex_ = 0;
    }

    void resetTurn() {
      evalPolicy_.reset();
      rootMoves_.clear();
    }

    static bool isCheckmateScore(int32_t score) noexcept {
      const int32_t absScore = std::abs(score);
      return evaluation::kCheckmateWinScore - 2 * kMaxDepthHardCutoff <= absScore && absScore <= evaluation::kCheckmateWinScore;
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      evalPolicy_.template markPiece<Add>(side, piece, square);
    }

    template <Side side>
    SearchResult search(BoardState& state, const SearchParam& param) {
      SearchContext context{.state = state, .maxDepth = param.maxDepth, .history = param.history, .nodes = 1, .probes = 0, .hits = 0, .plyReached = 0};

      constexpr NodeMeta meta(side, NodeMeta::NodeType::Internal, true, true, true);
      SearchFrame* frame = frames_.data();

      // Lazy move generation.
      if (rootMoves_.empty()) {
        state.template generateMoves<side>(rootMoves_);
        historyHeuristics_.clearKillers();
        ttTable_.age();
        drawHistoryIndex_ = std::max(0, static_cast<int>(context.history.size()) - 1 - state.getHalfmove());
        
        // Always call evaluate to sync NNUE state. Otherwise it desyncs in make/unmake.
        // TODO: Fix this brittle design.
        frame->clear();
        frame->staticScore = evalPolicy_.evaluate(state);
        frame->inCheck = context.state.template isInCheck<side>();
      }

      // Check checkmate and stalemate.
      const bool inCheck = *frame->inCheck;
      if (rootMoves_.empty()) [[unlikely]] {
        const int32_t score = inCheck ? evaluation::kCheckmateLossScore : evaluation::kStalemateScore;
        return SearchResult{.score = score, .move = Move{}, .pvLine = {}, .nodes = context.nodes, .searchingTime = {}, .probes = context.probes, .hits = context.hits, .plyReached = context.plyReached};
      }

      // Root node invariants: ply is guaranteed 0, prev1 and prev2 are empty.
      constexpr int ply = 0;
      TTProbeResult ttEval = probeTT<meta>(context, frame, context.maxDepth, 0, 0);

      // Aspiration window.
      const int32_t ttScore = (ttEval.type != TTProbeResult::Type::None ? ttEval.score : 0);
      AspirationWindow window(context.maxDepth, ttScore, aspirationWindow_, config_.aspirationWindowDivisor);

      StagedMovePicker movePicker(0, {}, {}, historyHeuristics_);
      movePicker.moves() = rootMoves_;

      // Search for the best move.
      for (;;) {
        Eval bestEval(window.alpha(), false);
        Move bestMove{};
        frame->pvLength = 0;

        movePicker.rePick();
        const size_t totalMoves = movePicker.size();
        for (size_t i = 0; i < totalMoves; ++i) {
          const auto [move, heuristicScore] = movePicker.pick<meta>(ttEval.move, state);

          const MoveUndo undo = state.template makeMove<side>(move, evalPolicy_);
          context.history.push(state.getHash());
          frame->move = move;

          const Eval eval = [&] {
            constexpr auto childMeta = meta.flip().setCanNullMove(true);
            const int extension = inCheck ? 1 : 0;
            const int depth = context.maxDepth;
            const int childDepth = depth - 1 + extension;

            // Full search if not enough depth to perform reduced depth search or on first move.
            if (i == 0 || depth < config_.lateMoveReductionDepth) {
              return searchChild<childMeta.setPV(true)>(context, frame + 1, childDepth, bestEval.score, window.beta());
            }

            // Late move reduction.
            // If reduction > 0, then LMR (narrow window + reduced depth).
            // If reduction == 0, then scout (narrow window).
            int reduction = 0;
            if (depth >= config_.lateMoveReductionDepth && !inCheck && (movePicker.isBadViolentMove() || (!move.isViolentMove() && !historyHeuristics_.isKillerMove(ply, move)))) {
              reduction = int(lateMoveReductions_[depth][i]);
              if (movePicker.isBadViolentMove()) {
                ++reduction;
              } else if (!move.isViolentMove()) {
                reduction += historyHeuristics_.getHistoryReduction(heuristicScore);
              }
              reduction = std::clamp(reduction, 0, childDepth - 1);
            }

            Eval result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth - reduction, bestEval.score, bestEval.score + 1);

            // If LMR improves alpha, research with scout.
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, frame + 1, childDepth, bestEval.score, bestEval.score + 1);
            }

            // If within window, full search.
            if (bestEval.score < result.score && result.score < window.beta()) {
              result = searchChild<childMeta.setPV(true)>(context, frame + 1, childDepth, bestEval.score, window.beta());
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
            if (bestEval.score >= window.beta()) {
              if (!move.isViolentMove()) {
                historyHeuristics_.template recordQuietCutoff<meta.side>(ply, context.maxDepth, movePicker.moves(), i);
              } else if (undo.capturedPiece != NoPiece) {
                historyHeuristics_.template recordCaptureCutoff<meta.side>(move.getMovedPiece(), move.to(), undo.capturedPiece, context.maxDepth);
              }
              ttEval.move = bestMove;
              break;
            }
          }
        }

        if (window.expand(bestEval.score)) {
          continue;
        }

        // Update TT and PV.
        if (!bestMove) {
          // Not reachable, because it can not fail lower on checkmate in 0.
          // But just in case...
          bestMove = movePicker.moves()[0];
        }
        bestEval.isUnreliable = false;
        storeTT<true, TranspositionTable::ScoreType::Exact>(context, frame, context.maxDepth, bestEval, bestMove, inCheck);
        const std::span<const Move> pvSpan{pvMoves(frame), frame->pvLength};
        return SearchResult{.score = bestEval.score, .move = bestMove, .pvLine = pvSpan, .nodes = context.nodes, .searchingTime = {}, .probes = context.probes, .hits = context.hits, .plyReached = context.plyReached};
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy, NegamaxSearchConfig, int32_t, size_t) -> NegamaxSearchPolicy<std::decay_t<EvalPolicy>>;
}
