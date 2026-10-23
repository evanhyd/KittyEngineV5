#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "searching_policy.h"
#include "transposition_table.h"
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

    struct NodeMeta {
      enum class NodeType {
        Internal,
        Quiescence,
      };

      const Side side;
      const NodeType type;
      const bool isPV;
      const bool canNullMove;

      consteval NodeMeta(Side side, NodeType type, bool isPV, bool canNullMove) noexcept
        : side(side), type(type), isPV(isPV), canNullMove(canNullMove) {}
      consteval NodeMeta flip() const noexcept { return NodeMeta{ getOtherSide(side), type, isPV, canNullMove }; }
      consteval NodeMeta enableNullMove() const noexcept { return NodeMeta{ side, type, isPV, true }; }
      consteval NodeMeta disableNullMove() const noexcept { return NodeMeta{ side, type, isPV, false }; }
      consteval NodeMeta toQuiescence() const noexcept { return NodeMeta{ side, NodeType::Quiescence, false, false }; }
      consteval NodeMeta setPV(bool pv) const noexcept { return NodeMeta{ side, type, pv, canNullMove }; }
      consteval bool isInternal() const noexcept { return type == NodeType::Internal; }
      consteval bool isQuiescence() const noexcept { return type == NodeType::Quiescence; }
    };

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
      std::array<Move, kMaxKillers> killerMoves;
      uint8_t pvLength;
      bool isFollowingPV;

      bool isKillerMove(Move move) const noexcept {
        return std::find(killerMoves.begin(), killerMoves.end(), move) != killerMoves.end();
      }
    };

    // Pruning Magic.
    static constexpr int kFutilityRequiredDepth = 3;
    static constexpr int32_t kFutilityMarginPerDepth = 100;
    static constexpr int kReverseFutilityRequiredDepth = 3;
    static constexpr int32_t kReverseFutilityMarginPerDepth = 100;

    // https://chessprogramming.org/Late_Move_Reductions#reduction-depth
    static inline const auto kLmrReductions = [] {
      std::array < std::array < uint8_t, MoveList{}.capacity() > , kMaxDepthHardCutoff + 1 > table{};
      for (size_t depth = 1; depth < table.size(); ++depth) {
        const double lnDepth = std::log(double(depth));
        for (size_t i = 1; i < table[depth].size(); ++i) {
          table[depth][i] = static_cast<uint8_t>(0.7844 + lnDepth * std::log(double(i)) / 2.4696);
        }
      }
      return table;
    }();

    constexpr int lateMovePruningThreshold(int depth) noexcept {
      return 3 + 2 * depth * depth;
    }

    // Move Ordering Priorities Magic.
    static constexpr int32_t kPrincipalVariationPriority = 30000;
    static constexpr int32_t kTranspositionPriority = 20000;
    static constexpr int32_t kPromotionPriority = 3000;
    static constexpr int32_t kUnderpromotionPriority = -1;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kKillerMovePriority = 99;
    static constexpr int32_t kMaxHistoryPriority = kKillerMovePriority - kMaxKillers;
    static constexpr std::array<int32_t, kPieceSize> kVictimPriority{1100, 1320, 1330, 1500, 1900, 5000};

    static constexpr int32_t capturePriority(Piece attacker, Piece victim) noexcept {
      return kVictimPriority[victim] - attacker;
    }

    EvalPolicy evalPolicy_;
    const int32_t aspirationWindow_;
    std::array<SearchStack, kMaxDepthHardCutoff + 1> searchStack_;
    // PV row at ply p has room for at most 65 - p moves.
    std::array<Move, kPvSlots> pvTable_;
    TranspositionTable ttTable_;
    std::array<std::array<std::array<int32_t, kSquareSize>, kSquareSize>, 2> historyTable_{};

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
      const size_t length = static_cast<size_t>((frame + 1)->pvLength) + 1;
      Move* line = pvMoves(frame);
      line[0] = move;
      std::copy_n(pvMoves(frame + 1), (frame + 1)->pvLength, line + 1);
      frame->pvLength = static_cast<uint8_t>(length);
    }

    bool isCheckmateScore(int32_t score) const noexcept {
      return evaluation::kCheckmateWinScore - std::abs(score) <= kMaxDepthHardCutoff;
    }

    // Convert between root-relative and node-relative checkmate distance penalties.
    int32_t normalizeScore(int32_t score, int ply) const noexcept {
      if (!isCheckmateScore(score)) {
        return score;
      }
      return score < 0 ? score - ply : score + ply;
    }

    int32_t denormalizeScore(int32_t score, int ply) const noexcept {
      return normalizeScore(score, -ply);
    }

    void storeTT(const SearchContext& context, SearchStack* frame, int depth, Eval eval,
                 TranspositionTable::ScoreType scoreType, Move bestMove) noexcept {
      if (!eval.isUnreliable) {
        ttTable_.put(TranspositionTable::Entry{
          .key = context.state.getHash(),
          .depth = static_cast<int16_t>(depth),
          .scoreType = scoreType,
          .bestMove = bestMove,
          .score = normalizeScore(eval.score, getPly(frame)),
        });
      }
    }

    std::optional<Eval> checkDraw(const SearchContext& context, int ply) const noexcept {
      const auto halfmove = context.state.getHalfmove();
      if (halfmove < 4) {
        return std::nullopt;
      }
      if (halfmove >= 100) {
        return Eval(0, true);
      }
      const auto hBegin = context.positionHistory.rbegin();
      const auto hEnd = hBegin + std::min(context.positionHistory.size(), static_cast<size_t>(halfmove) + 1);
      if (const auto it = std::find(hBegin + 1, hEnd, context.state.getHash()); it != hEnd) {
        const bool inGameHistory = int(std::distance(hBegin, it)) >= ply;
        return Eval(0, !inGameHistory);
      }
      return std::nullopt;
    }

    template <NodeMeta meta>
    std::optional<Eval> probeTT(const SearchContext& context, SearchStack* frame, int depth,
                                int32_t alpha, int32_t beta, Move& ttMove) noexcept {
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return std::nullopt;
      }
      ttMove = ttEntry->bestMove;
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
      int32_t& score = historyTable_[side][move.from()][move.to()];
      score = std::clamp(score + reward, 0, kMaxHistoryPriority);
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

    template <Side ally>
    int32_t staticExchangeEvaluate(const BoardState& state, Move move) const noexcept {
      constexpr Side enemy = getOtherSide(ally);
      const Square square = move.to();
      Bitboard occupancy = moveSquare(state.getOccupancy(White) | state.getOccupancy(Black), move.from(), square);
      if (move.isEnpassant()) {
        occupancy = unsetSquare(occupancy, ally == White ? squareDown(square) : squareUp(square));
      }
      const Piece promotion = move.getPromotedPieceType();
      const Piece attacker = promotion == NoPiece ? move.getMovedPiece() : promotion;
      const Piece victim = move.isCapture() ? state.getPieceAt<enemy>(square) :
        (move.isEnpassant() ? Pawn : NoPiece);
      std::array<Piece, kSideSize> pieceIndex{Pawn, Pawn};

      auto popLVA = [&](Side side) -> Piece {
        for (Piece piece = pieceIndex[side]; piece < kPieceSize; ++piece) {
          Bitboard attackers = 0;
          switch (piece) {
          case Pawn: attackers = (side == White ? getAttack<Pawn, Black>(square) : getAttack<Pawn, White>(square)) & state.getPieces(side, Pawn); break;
          case Knight: attackers = getAttack<Knight>(square) & state.getPieces(side, Knight); break;
          case Bishop: attackers = getAttack<Bishop>(square, occupancy) & state.getPieces(side, Bishop); break;
          case Rook: attackers = getAttack<Rook>(square, occupancy) & state.getPieces(side, Rook); break;
          case Queen: attackers = getAttack<Queen>(square, occupancy) & state.getPieces(side, Queen); break;
          case King: attackers = getAttack<King>(square) & state.getPieces(side, King); break;
          default: break;
          }
          attackers &= occupancy;
          if (attackers) {
            occupancy = unsetSquare(occupancy, peekPiece(attackers));
            return piece;
          }
          // Pawn and knight attacks cannot be uncovered by removing blockers.
          if (piece < Bishop) {
            pieceIndex[side] = piece + 1;
          }
        }
        return NoPiece;
      };

      // https://chess.stackexchange.com/questions/44853/whats-the-maximum-number-of-pieces-attacking-a-single-square
      SmallVec<int32_t, 32> gains;
      gains.push(victim == NoPiece ? 0 : capturePriority(attacker, victim));
      Piece exchangedPiece = attacker;
      Side side = enemy;

      while (gains.size() < gains.capacity()) {
        const Piece nextAttacker = popLVA(side);
        if (nextAttacker == NoPiece) {
          break;
        }
        gains.push(capturePriority(nextAttacker, exchangedPiece));
        exchangedPiece = nextAttacker;
        side = getOtherSide(side);
      }

      int32_t reply = 0;
      for (size_t i = gains.size() - 1; i > 0; --i) {
        reply = std::max(0, gains[i] - reply); // Each side may decline to recapture.
      }
      return gains.front() - reply;
    }

    int32_t staticExchangeEvaluate(const BoardState& state, Move move) const noexcept {
      return state.getSide() == White ? staticExchangeEvaluate<White>(state, move) : staticExchangeEvaluate<Black>(state, move);
    }

    template <NodeMeta meta>
    class StageMoveSelector {

    };

    template <NodeMeta meta>
    void sortMoves(const BoardState& state, MoveList& moves, Move ttMove, Move pvMove, const decltype(SearchStack::killerMoves)& killerMoves) const noexcept {
      if (moves.size() <= 1) {
        return;
      }
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
          if constexpr (meta.isInternal()) {
            if (move == pvMove) {
              priority += kPrincipalVariationPriority;
            }
          }
          // A TT or PV move already outranks every ordinary move.
          if (priority != 0) {
            return PriorityMove{priority, move};
          }
        }

        if (move.isViolentMove()) {
          priority += staticExchangeEvaluate<meta.side>(state, move);
          const Piece promotion = move.getPromotedPieceType();
          if (promotion != NoPiece) {
            priority += promotion == Queen ? kPromotionPriority : kUnderpromotionPriority;
          }
        } else if constexpr (meta.isInternal()) {
          // Killer move heuristics.
          for (size_t i = 0; i < killerMoves.size() && killerMoves[i] != Move{}; ++i) {
            if (move == killerMoves[i]) {
              priority += kKillerMovePriority - int32_t(i);
            }
          }

          // History heuristics.
          priority += historyTable_[meta.side][move.from()][move.to()];

          if (move.isCastling()) {
            priority += kCastlingPriority;
          }
        }

        return PriorityMove{ priority, move };
      });

      // Sort the moves based on priority, then map back to the original moves list.
      std::ranges::sort(scoredMoves, std::greater<>{}, &PriorityMove::priority);
      std::ranges::transform(scoredMoves, moves.begin(), &PriorityMove::move);
    }

    //////////////////
    // SEARCH CHILD //
    //////////////////
    template <NodeMeta meta>
    Eval searchChild(SearchContext& context, SearchStack* childFrame, const int childDepth, const int32_t childAlpha, const int32_t childBeta) {
      Eval result = searchInternal<meta>(context, childFrame, childDepth, -childBeta, -childAlpha);
      result.score = -result.score;
      return result;
    }

    template <NodeMeta meta>
    Eval searchInternal(SearchContext& context, SearchStack* frame, const int depth,
                        const int32_t alpha, const int32_t beta) {
      ++context.searchedNodes;
      const int ply = getPly(frame);
      Move pvMove{};
      Move ttMove{};

      // Clear old PV written by sibling node and load expected PV move.
      if constexpr (meta.isInternal()) {
        frame->pvLength = 0;
        if (frame->isFollowingPV && context.pvLine.size() > ply) {
          pvMove = context.pvLine[ply];
        }
      }

      if constexpr (!meta.isQuiescence()) {
        if (auto drawEval = checkDraw(context, ply)) {
          return *drawEval;
        }
        if (auto ttEval = probeTT<meta>(context, frame, depth, alpha, beta, ttMove); ttEval) {
          return *ttEval;
        }
        if (depth == 0) {
          --context.searchedNodes;
          return searchInternal<meta.toQuiescence()>(context, frame, 0, alpha, beta);
        }
      } else {
        if (ply == kMaxDepthHardCutoff) {
          return Eval(evalPolicy_.evaluate(context.state), false);
        }
      }

      MoveList moves;
      Eval bestEval(alpha, false);

      // Must be internal node. Disable on PV, in check, or with low material (zugzwang risk).
      std::optional<int32_t> staticEvalScore;

      const bool inCheck = context.state.template isInCheck<meta.side>();
      if constexpr (meta.isInternal()) {
        // Checkmate or draw.
        context.state.template generateMoves<meta.side>(moves);
        if (moves.empty()) {
          return Eval(inCheck ? evaluation::kCheckmateLossScore + ply : evaluation::kStalemateScore, false);
        }

        if constexpr (!meta.isPV) {
          if (!inCheck && context.state.template countMMPiece<meta.side>() >= 2) {
            if (depth <= kReverseFutilityRequiredDepth && !isCheckmateScore(beta)) {
              staticEvalScore = evalPolicy_.evaluate(context.state);

              // Reverse Futility Pruning (Static Null Move Pruning).
              if (*staticEvalScore - depth * kReverseFutilityMarginPerDepth >= beta) {
                const Eval eval(beta, false);
                storeTT(context, frame, depth, eval, TranspositionTable::ScoreType::LowerBound, ttMove);
                return eval;
              }
            }

            // Null Move Pruning. Disable near leaves. Disable consecutive null moves.
            if (depth >= 2 && meta.canNullMove) {
              const int skipped = std::max(1, depth / 2);
              MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
              context.positionHistory.push(context.state.getHash());
              (frame + 1)->isFollowingPV = false;
              Eval eval = searchInternal<meta.flip().disableNullMove()>(context, frame + 1, depth - skipped, -beta, -beta + 1);
              eval.score = -eval.score;
              context.positionHistory.pop();
              context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);

              if (eval.score >= beta) {
                if (isCheckmateScore(eval.score)) {
                  eval.score = beta;
                }
                storeTT(context, frame, depth, eval, TranspositionTable::ScoreType::LowerBound, ttMove);
                return eval;
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
      [[maybe_unused]] Move bestMove = moves[0];
      int deadQuietMoveCount = 0;
      const int lateMovePruningRequiredMove = lateMovePruningThreshold(depth);

      // Explore moves.
      for (size_t i = 0; i < moves.size(); ++i) {
        const Move move = moves[i];
        const MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);

        if constexpr (meta.isInternal() && !meta.isPV) {
          const bool isDeadQuiet = (
            i != 0 &&
            !inCheck && 
            !move.isViolentMove() && 
            !frame->isKillerMove(move) && 
            context.state.template countMMPiece<meta.side>() >= 2 && 
            !context.state.template isInCheck<getOtherSide(meta.side)>());

          if (isDeadQuiet) {
            // Late Move Pruning
            ++deadQuietMoveCount;
            if (deadQuietMoveCount >= lateMovePruningRequiredMove) {
              context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);
              continue;
            }

            // Futility Pruning. 
            if (depth <= kFutilityRequiredDepth) { // 100 cycles

              // Lazy-init the static evaluation.
              if (!staticEvalScore) {
                staticEvalScore = evalPolicy_.evaluate(context.state); // 1200 cycles.
              }

              // Prune away hopeless quiet move.
              if (*staticEvalScore + kFutilityMarginPerDepth * depth <= bestEval.score) {
                context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);
                continue;
              }
            }
          }
        }

        context.positionHistory.push(context.state.getHash());
        //ttTable_.prefetch(context.state.getHash());

        if constexpr (meta.isInternal()) {
          (frame + 1)->isFollowingPV = frame->isFollowingPV && move == pvMove;
        }

        const Eval eval = [&] {
          if constexpr (meta.isInternal()) {
            constexpr auto childMeta = meta.flip().enableNullMove();

            // Disable narrow window on PV and near leaf.
            if (i == 0 || depth == 1) {
              return searchChild<childMeta.setPV(meta.isPV)>(context, frame + 1, depth - 1, bestEval.score, beta);
            }

            // Late move reduction.
            int reduction = 0;
            if (depth >= 3 && !inCheck && !move.isViolentMove()) {
              reduction = int(kLmrReductions[depth][i]);
            }

            // LMR search (narrow window + reduced depth).
            Eval result = searchChild<childMeta.setPV(false)>(context, frame + 1, depth - 1 - reduction, bestEval.score, bestEval.score + 1);

            // PV search if alpha improves (narrow window).
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, frame + 1, depth - 1, bestEval.score, bestEval.score + 1);
            }

            // Full window search if within window.
            if (bestEval.score < result.score && result.score < beta) {
              result = searchChild<childMeta.setPV(true)>(context, frame + 1, depth - 1, bestEval.score, beta);
            }
            return result;
          } else {
            return searchChild<meta.flip().setPV(false)>(context, frame + 1, depth - 1, bestEval.score, beta);
          }
        }();

        context.positionHistory.pop();
        context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);

        if (eval.score >= beta) {
          if constexpr (!meta.isQuiescence()) {
            storeTT(context, frame, depth, eval, TranspositionTable::ScoreType::LowerBound, move);
            if constexpr (meta.isInternal()) {
              recordQuietCutoff(meta.side, frame, moves, i, depth);
            }
          }
          return eval;
        }
        if (eval.score > bestEval.score) {
          bestEval = eval;
          if constexpr (!meta.isQuiescence()) {
            bestMove = move;
            if constexpr (meta.isInternal()) {
              acceptChildPVLine(frame, bestMove);
            }
          }
        }
      }
      
      if constexpr (!meta.isQuiescence()) {
        const auto scoreType = (bestEval.score <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact);
        storeTT(context, frame, depth, bestEval, scoreType, bestMove);
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

      // PV table.
      Move pvMove{};
      if (!context.pvLine.empty()) {
        pvMove = context.pvLine.front();
      }
      searchStack_[0].pvLength = 0;

      // Lookup transposition table for move ordering.
      Move ttMove{};
      if (auto ttEntry = ttTable_.get(state.getHash())) {
        ttMove = ttEntry->bestMove;
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
      sortMoves<meta>(state, moves, ttMove, pvMove, searchStack_[0].killerMoves);

      // Set up aspiration window.
      const auto expandWindow = [](int& failTime, int32_t window) noexcept {
        ++failTime;
        switch (failTime) {
        case 1: return 3 * window;
        case 2: return 10 * window;
        default: return evaluation::kCheckmateWinScore;
        }
      };
      int failLowCount = 0;
      int failHighCount = 0;
      auto [alpha, beta] = context.maxDepth <= 4
        ? std::pair{evaluation::kCheckmateLossScore, evaluation::kCheckmateWinScore}
        : std::pair{param.pastEval - aspirationWindow_, param.pastEval + aspirationWindow_};

      const bool inCheck = state.isInCheck<ally>();

      // Search for the best move.
      for (;;) {
        Eval bestEval(alpha, false);
        Move bestMove = moves[0];

        for (size_t i = 0; i < moves.size(); ++i) {
          const Move move = moves[i];
          const MoveUndo undo = state.makeMove<ally>(move, evalPolicy_);
          param.positionHistory.push(state.getHash());
          searchStack_[1].isFollowingPV = move == pvMove;

          const int depth = context.maxDepth;

          const Eval eval = [&] {
            constexpr auto childMeta = meta.flip().enableNullMove();

            // Disable narrow window on PV and near leaf.
            if (i == 0 || depth == 1) {
              return searchChild<childMeta.setPV(true)>(context, searchStack_.data() + 1, depth - 1, bestEval.score, beta);
            }

            // Late move reduction.
            int reduction = 0;
            if (depth >= 3 && !inCheck && !move.isViolentMove()) {
              reduction = int(kLmrReductions[depth][i]);
            }

            // LMR search (narrow window + reduced depth).
            Eval result = searchChild<childMeta.setPV(false)>(context, searchStack_.data() + 1, depth - 1 - reduction, bestEval.score, bestEval.score + 1);

            // PV search (narrow window).
            if (reduction > 0 && result.score > bestEval.score) {
              result = searchChild<childMeta.setPV(false)>(context, searchStack_.data() + 1, depth - 1, bestEval.score, bestEval.score + 1);
            }

            // Full window search.
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
              std::swap(moves[0], moves[i]);
              break;
            }
          }
        }

        if (bestEval.score >= beta) {
          // Fail-high, the position is better than expected, increase beta and re-search.
          beta = std::min(evaluation::kCheckmateWinScore, beta + expandWindow(failHighCount, aspirationWindow_));
          continue;
        } else if (bestEval.score <= alpha) {
          // Fail-low, the position is worse than expected, decrease alpha and re-search.
          alpha = std::max(evaluation::kCheckmateLossScore, alpha - expandWindow(failLowCount, aspirationWindow_));
          continue;
        }

        // Update TT and PV.
        storeTT(context, searchStack_.data(), context.maxDepth, bestEval, TranspositionTable::ScoreType::Exact, bestMove);
        param.pvLine.resize(searchStack_[0].pvLength);
        std::copy_n(pvMoves(searchStack_.data()), searchStack_[0].pvLength, param.pvLine.begin());
        return makeResult(bestEval.score, bestMove);
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
