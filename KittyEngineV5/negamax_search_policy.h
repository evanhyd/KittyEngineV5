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
    struct NodeMeta {
      enum class NodeType {
        Root,
        Internal,
        NullMove,
        Quiescence,
      };

      const Side side;
      const NodeType type;
      consteval NodeMeta(Side side, NodeType type) noexcept : side(side), type(type) {}
      consteval NodeMeta flip() const noexcept { return NodeMeta{ getOtherSide(side), type }; }
      consteval NodeMeta toInternal() const noexcept { return NodeMeta{ side, NodeType::Internal }; }
      consteval NodeMeta toNullMove() const noexcept { return NodeMeta{ side, NodeType::NullMove }; }
      consteval NodeMeta toQuiescence() const noexcept { return NodeMeta{ side, NodeType::Quiescence }; }
      consteval bool isRoot() const noexcept { return type == NodeType::Root; }
      consteval bool isInternal() const noexcept { return type == NodeType::Internal; }
      consteval bool isNullMove() const noexcept { return type == NodeType::NullMove; }
      consteval bool isQuiescence() const noexcept { return type == NodeType::Quiescence; }
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

    static constexpr size_t kMaxKillers = 3;
    struct SearchStack {
      std::array<Move, kMaxKillers> killerMoves;
      uint8_t pvLength;
      bool isFollowingPV;
    };
    static constexpr size_t kPvSlots = (kMaxDepthHardCutoff + 1) * (kMaxDepthHardCutoff + 2) / 2;

    // Magic constant.
    // static constexpr int32_t kFutilityMovePriority = 0;
    static constexpr int32_t kPrincipalVariationPriority = 10000;
    static constexpr int32_t kTranspositionPriority = 20000;
    static constexpr int32_t kPromotionPriority = 3000;
    static constexpr int32_t kUnderpromotionPriority = 50;
    static constexpr int32_t kCastlingPriority = 151;
    static constexpr int32_t kKillerMovePriority = 99;
    static constexpr int32_t kMaxHistoryPriority = kKillerMovePriority - kMaxKillers;
    static inline const auto kLmrReductions = [] {
      std::array<std::array<uint8_t, MoveList{}.capacity()>, kMaxDepthHardCutoff + 1> table{};
      for (size_t depth = 1; depth < table.size(); ++depth) {
        const double lnDepth = std::log(double(depth));
        for (size_t i = 1; i < table[depth].size(); ++i) {
          table[depth][i] = static_cast<uint8_t>(0.7844 + lnDepth * std::log(double(i)) / 2.4696);
        }
      }
      return table;
    }();
    // 1000 + victim value - attacker index; the king uses a 4000-point SEE sentinel.
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

    bool isCheckmateScore(int32_t score) const noexcept {
      return -evaluation::kCheckmateScore - std::abs(score) <= kMaxDepthHardCutoff;
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

    void acceptChildPVLine(SearchStack* frame, Move move) noexcept {
      const size_t length = static_cast<size_t>((frame + 1)->pvLength) + 1;
      Move* line = pvMoves(frame);
      line[0] = move;
      std::copy_n(pvMoves(frame + 1), (frame + 1)->pvLength, line + 1);
      frame->pvLength = static_cast<uint8_t>(length);
    }

    template <NodeMeta meta>
    void storeTT(const SearchContext& context, SearchStack* frame, int depth, Eval eval,
                 TranspositionTable::ScoreType scoreType, Move bestMove) noexcept {
      if constexpr (meta.isRoot() || meta.isInternal() || meta.isNullMove()) {
        if (!eval.historyDependent) {
          ttTable_.put(TranspositionTable::Entry{
            .key = context.state.getHash(),
            .depth = static_cast<int16_t>(depth),
            .scoreType = scoreType,
            .bestMove = bestMove,
            .score = normalizeScore(eval.score, getPly(frame)),
          });
        }
      }
    }

    std::optional<Eval> checkDraw(const SearchContext& context, int ply) const noexcept {
      const auto hBegin = context.positionHistory.rbegin();
      const auto hEnd = hBegin + std::min(context.positionHistory.size(),
                                          static_cast<size_t>(context.state.getHalfmoveClock()) + 1);
      if (const auto it = std::find(hBegin + 1, hEnd, context.state.getHash()); it != hEnd) {
        const bool inGameHistory = int(std::distance(hBegin, it)) >= ply;
        return Eval(0, !inGameHistory);
      }
      if (context.state.getHalfmoveClock() >= 100) {
        return Eval(0, false);
      }
      return std::nullopt;
    }

    template <NodeMeta meta>
    std::optional<Eval> probeTT(const SearchContext& context, SearchStack* frame, int depth,
                                int32_t alpha, int32_t beta, Move pvMove, Move& ttMove) noexcept {
      auto ttEntry = ttTable_.get(context.state.getHash());
      if (!ttEntry) {
        return std::nullopt;
      }
      ttMove = ttEntry->bestMove;
      if (ttEntry->depth < depth) {
        return std::nullopt;
      }

      const int ply = getPly(frame);
      const int32_t score = denormalizeScore(ttEntry->score, ply);
      if (ttEntry->scoreType == TranspositionTable::ScoreType::Exact) {
        if constexpr (meta.isInternal()) {
          if (ttEntry->bestMove == pvMove) {
            const size_t length = context.pvLine.size() - ply;
            std::copy(context.pvLine.begin() + ply, context.pvLine.end(), pvMoves(frame));
            frame->pvLength = static_cast<uint8_t>(length);
          } else {
            pvMoves(frame)[0] = ttEntry->bestMove;
            frame->pvLength = 1;
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

    void recordKiller(decltype(SearchStack::killerMoves)& killers, Move move) noexcept {
      const auto it = std::find(killers.begin(), killers.end(), move);
      const size_t slot = it == killers.end() ? killers.size() - 1 : static_cast<size_t>(it - killers.begin());
      for (size_t i = slot; i > 0; --i) {
        killers[i] = killers[i - 1];
      }
      killers[0] = move;
    }

    void recordHistory(Side side, Move move, int32_t reward) noexcept {
      int32_t bonus = historyTable_[side][move.from()][move.to()];
      historyTable_[side][move.from()][move.to()] = std::clamp(bonus + reward, 0, kMaxHistoryPriority);
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

    int32_t staticExchangeEvaluate(const BoardState& state, Move move) const noexcept {
      const Square square = move.to();
      Bitboard occupancy = moveSquare(state.getOccupancy(White) | state.getOccupancy(Black), move.from(), square);
      if (move.isEnpassant()) {
        occupancy = unsetSquare(occupancy, state.getSide() == White ? squareDown(square) : squareUp(square));
      }
      const Piece promotion = move.getPromotedPieceType();
      const Piece attacker = promotion == NoPiece ? move.getMovedPiece() : promotion;
      const Piece victim = move.isCapture() ? state.getPieceAt(square).piece :
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
      Side side = getOtherSide(state.getSide());

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

    template <NodeMeta meta>
    void sortMoves(const BoardState& state, MoveList& moves, Move ttMove, Move pvMove, const decltype(SearchStack::killerMoves)& killerMoves) const noexcept {
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
          // A TT or PV move already outranks every ordinary move.
          if (priority != 0) {
            return PriorityMove{priority, move};
          }
        }

        if (move.isViolentMove()) {
          priority += staticExchangeEvaluate(state, move);
          const Piece promotion = move.getPromotedPieceType();
          if (promotion != NoPiece) {
            priority += promotion == Queen ? kPromotionPriority : kUnderpromotionPriority;
          }
        } else if constexpr (!meta.isQuiescence()) {
          if constexpr (!meta.isNullMove()) {
            // Killer move heuristics.
            for (size_t i = 0; i < killerMoves.size() && killerMoves[i] != Move{}; ++i) {
              if (move == killerMoves[i]) {
                priority += kKillerMovePriority - int32_t(i);
              }
            }

            // History heuristics.
            priority += historyTable_[meta.side][move.from()][move.to()];
          }
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

    /////////////////////
    // SEARCH INTERNAL //
    /////////////////////
    template <NodeMeta meta>
    Eval searchInternal(SearchContext& context, SearchStack* frame, int depth,
                        const int32_t alpha, const int32_t beta) {
      ++context.searchedNodes;
      const int ply = getPly(frame);
      frame->pvLength = 0;
      Move pvMove{};
      Move ttMove{};

      // Clear the old PV written by a sibling node and load the expected PV move.
      if constexpr (meta.isInternal()) {
        if (frame->isFollowingPV && context.pvLine.size() > ply) {
          pvMove = context.pvLine[ply];
        }
      }

      if (auto drawEval = checkDraw(context, ply)) {
        return *drawEval;
      }

      if constexpr (!meta.isQuiescence()) {
        if (auto ttEval = probeTT<meta>(context, frame, depth, alpha, beta, pvMove, ttMove); ttEval) {
          return *ttEval;
        }
        if (depth == 0) {
          --context.searchedNodes;
          return searchInternal<meta.toQuiescence()>(context, frame, 0, alpha, beta);
        }
      } else if (ply == kMaxDepthHardCutoff) {
        return Eval(evalPolicy_.evaluate(context.state), false);
      }

      MoveList moves;
      Eval bestEval(alpha, false);

      const bool inCheck = context.state.template isInCheck<meta.side>();
      if constexpr (meta.isInternal() || meta.isNullMove()) {
        context.state.template generateMoves<meta.side>(moves);
        if (moves.empty()) {
          return Eval(inCheck ? evaluation::kCheckmateScore + ply : evaluation::kStalemateScore, false);
        }

        // Null Move Pruning: disabled near leaves, on PV, in check, or with low material (zugzwang risk).
        if constexpr (meta.isInternal()) {
          if (depth > 1 && !frame->isFollowingPV && !inCheck && context.state.template countMMPiece<meta.side>() >= 2) {
            const int skipped = std::max(1, depth / 2);
            MoveUndo undo = context.state.template makeMove<meta.side, true>(Move{}, evalPolicy_);
            context.positionHistory.push(context.state.getHash());
            (frame + 1)->isFollowingPV = false;
            Eval eval = searchInternal<meta.flip().toNullMove()>(context, frame + 1, depth - skipped, -beta, -beta + 1);
            context.positionHistory.pop();
            context.state.template unmakeMove<meta.side, true>(Move{}, undo, evalPolicy_);
            eval.score = -eval.score;
            if (eval.score >= beta) {
              if (isCheckmateScore(eval.score)) {
                eval.score = beta;
              }
              storeTT<meta>(context, frame, depth, eval, TranspositionTable::ScoreType::LowerBound, ttMove);
              return eval;
            }
          }
        }
      } else {
        // Quiescence search: no stand-pat when in check.
        // Allow quiescence to ignore stalemate check, since it is too rare.
        if (inCheck) {
          context.state.template generateMoves<meta.side>(moves);
          if (moves.empty()) {
            return Eval(evaluation::kCheckmateScore + ply, false);
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
      const int childDepth = meta.isQuiescence() ? 0 : depth - 1;

      // Explore moves.
      for (size_t i = 0; i < moves.size(); ++i) {
        Move move = moves[i];
        MoveUndo undo = context.state.template makeMove<meta.side>(move, evalPolicy_);
        context.positionHistory.push(context.state.getHash());
        (frame + 1)->isFollowingPV = meta.isInternal() && frame->isFollowingPV && move == pvMove;

        const auto searchChild = [&](int childDepth, int32_t childAlpha, int32_t childBeta) {
          Eval result = searchInternal<meta.flip()>(context, frame + 1, childDepth, -childBeta, -childAlpha);
          result.score = -result.score;
          return result;
        };
        Eval eval = [&]() -> Eval {
          // The first move and quiescence/null-move must be searched in full.
          if (!meta.isInternal() || i == 0) {
            return searchChild(childDepth, bestEval.score, beta);
          }

          // Late move reduction.
          // https://chessprogramming.org/Late_Move_Reductions#reduction-depth
          int reduction = 0;
          if (depth >= 3 && !move.isViolentMove() && !inCheck) {
            reduction = std::min<int>(kLmrReductions[depth][i], childDepth - 1);
          }

          // Scout search (reduced depth + narrow window).
          Eval result = searchChild(childDepth - reduction, bestEval.score, bestEval.score + 1);

          // PV search (narrow window).
          if (reduction > 0 && result.score > bestEval.score) {
            result = searchChild(childDepth, bestEval.score, bestEval.score + 1);
          }

          // Full window search.
          if (bestEval.score < result.score && result.score < beta) {
            result = searchChild(childDepth, bestEval.score, beta);
          }
          return result;
        }();

        context.positionHistory.pop();
        context.state.template unmakeMove<meta.side>(move, undo, evalPolicy_);

        if (eval.score >= beta) {
          storeTT<meta>(context, frame, depth, eval, TranspositionTable::ScoreType::LowerBound, move);
          if constexpr (meta.isInternal()) {
            recordQuietCutoff(meta.side, frame, moves, i, depth);
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
      
      const auto scoreType = (bestEval.score <= alpha ? TranspositionTable::ScoreType::UpperBound : TranspositionTable::ScoreType::Exact);
      storeTT<meta>(context, frame, depth, bestEval,  scoreType, bestMove);
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
      searchStack_[0].pvLength = 0;

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
        ttTable_.age();
      }
      MoveList& moves = *param.rootMoves;
      if (moves.empty()) {
        context.pvLine.clear();
        const int32_t score = state.isInCheck<ally>() ? evaluation::kCheckmateScore : evaluation::kStalemateScore;
        return makeResult(score, std::nullopt);
      }
      sortMoves<meta>(state, moves, ttMove, pvMove, searchStack_[0].killerMoves);

      // Set up aspiration window.
      const auto expandWindow = [](int& failTime, int32_t window) noexcept {
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

        for (size_t i = 0; i < moves.size(); ++i) {
          const Move move = moves[i];
          const MoveUndo undo = state.makeMove<ally>(move, evalPolicy_);
          param.positionHistory.push(state.getHash());
          searchStack_[1].isFollowingPV = move == pvMove;
          Eval eval = searchInternal<meta.flip().toInternal()>(context, searchStack_.data() + 1,
                                                                context.maxDepth - 1, -beta, -bestEval.score);
          eval.score = -eval.score;
          param.positionHistory.pop();
          state.unmakeMove<ally>(move, undo, evalPolicy_);
          if (eval.score > bestEval.score) {
            bestEval = eval;
            bestMove = move;
            if (bestEval.score >= beta) {
              recordQuietCutoff(meta.side, searchStack_.data(), moves, i, context.maxDepth);
              std::swap(moves[0], moves[i]);
              break;
            }
            acceptChildPVLine(searchStack_.data(), bestMove);
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
        storeTT<meta>(context, searchStack_.data(), context.maxDepth, bestEval, TranspositionTable::ScoreType::Exact, bestMove);
        param.pvLine.resize(searchStack_[0].pvLength);
        std::copy_n(pvMoves(searchStack_.data()), searchStack_[0].pvLength, param.pvLine.begin());
        return makeResult(bestEval.score, bestMove);
      }
    }
  };

  template <typename EvalPolicy>
  NegamaxSearchPolicy(EvalPolicy) -> NegamaxSearchPolicy<EvalPolicy>;
}
