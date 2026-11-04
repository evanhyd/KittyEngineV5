#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "nnue_mlp.h"
#include "nnue_serializer.h"
#include "searching_policy.h"
#include "small_vec.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace bb::evaluation {
  struct NnueLayout {
    static constexpr uint32_t kPieceTypes = 6; // 6 piece types (Pawn, Knight, Bishop, Rook, Queen, King)
    static constexpr uint32_t kKingPlaneStride = 2 * kPieceTypes * kSquareSize; // 768 inputs per king square
    static constexpr uint32_t kPiecePlaneStride = kSquareSize;                  // 64 squares per piece plane
    static constexpr uint32_t kInputSize = nnue::NnueModel::kInputSize;         // 49,152 total sparse inputs
    using FeatureList = SmallVec<uint32_t, kSquareSize>;

    static constexpr Square orient(Side perspective, Square square) noexcept {
      if (perspective == White) {
        return square;
      }
      return Square(square ^ kRankFlipMask);
    }

    static constexpr uint32_t kingBase(Side perspective, Square kingSquare) noexcept {
      const uint32_t orientedKing = orient(perspective, kingSquare);
      return orientedKing * kKingPlaneStride;
    }

    static constexpr uint32_t pieceOffset(Side perspective, Side pieceSide, Piece piece, Square pieceSquare) noexcept {
      const uint32_t relativeSide = pieceSide ^ perspective;
      const uint32_t orientedPiece = orient(perspective, pieceSquare);
      const uint32_t pieceIdx = relativeSide * kPieceTypes + piece;
      return pieceIdx * kPiecePlaneStride + orientedPiece;
    }

    // Input Index: [own king square: 64] * 768 + [piece index: 12] * 64 + [piece square: 64]
    static constexpr uint32_t pieceIndex(Side perspective, Side pieceSide, Piece piece, Square kingSquare, Square pieceSquare) noexcept {
      return kingBase(perspective, kingSquare) + pieceOffset(perspective, pieceSide, piece, pieceSquare);
    }

    static Square kingSquare(const BoardState& state, Side perspective) {
      return peekPiece(state.getPieces(perspective, King));
    }

    static FeatureList encode(const BoardState& state, Side perspective,
                              Square kingSquare) {
      FeatureList features;
      for (Side relativeSide = White; relativeSide <= Black; ++relativeSide) {
        const Side pieceSide = perspective ^ relativeSide;
        for (Piece piece = Pawn; piece <= King; ++piece) {
          for (Bitboard pieces = state.getPieces(pieceSide, piece);
               pieces != 0;
               pieces = popPiece(pieces)) {
            features.push(pieceIndex(perspective, pieceSide, piece, kingSquare,
                                     peekPiece(pieces)));
          }
        }
      }
      return features;
    }
  };


  class NNUEEvaluationPolicy {
    // A move changes at most four features per ply (e.g. castling moves both king and rook).
    static constexpr size_t kHistoryCapacity = 4 * searching::kMaxDepthHardCutoff;

    struct PieceChange {
      Side side;
      Piece piece;
      Square square;
      bool add;
    };

    std::unique_ptr<nnue::NnueModel> model_;
    SmallVec<PieceChange, kHistoryCapacity> history_;
    size_t applied_ = 0;
    std::array<Square, 2> cachedKingSquares_{NoSquare, NoSquare};

    template <bool Add>
    void applyChange(Side perspective, Side side, Piece piece, Square square) noexcept {
      const uint32_t index = NnueLayout::pieceIndex(
        perspective, side, piece, cachedKingSquares_[perspective], square);
      model_->feedIncremental<Add>(index, perspective);
    }

    void rebuild(const BoardState& state, Side perspective, Square kingSquare) {
      const auto features = NnueLayout::encode(state, perspective, kingSquare);
      model_->feedFull({features.begin(), features.size()}, perspective);
      cachedKingSquares_[perspective] = kingSquare;
    }

  public:
    explicit NNUEEvaluationPolicy(const std::filesystem::path& filename)
      : model_(nnue::NnueSerializer::load(filename)) {}

    void reset() noexcept {
      history_.clear();
      applied_ = 0;
      cachedKingSquares_.fill(NoSquare);
      model_->clearAccumulators();
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      if (!model_->hasAccumulator(White)) {
        return;
      }

      if (piece == King) {
        cachedKingSquares_[side] = NoSquare;
      }

      if (!history_.empty()) {
        const PieceChange& top = history_.back();
        if (top.side == side && top.piece == piece && top.square == square && top.add != Add) {
          if (history_.size() <= applied_) {
            for (Side perspective = White; perspective <= Black; ++perspective) {
              if (cachedKingSquares_[perspective] != NoSquare) {
                applyChange<Add>(perspective, side, piece, square);
              }
            }
            --applied_;
          }
          history_.pop();
          return;
        }
      }

      if (history_.size() == kHistoryCapacity) {
        reset();
        return;
      }

      history_.push({side, piece, square, Add});
    }

    int32_t evaluate(const BoardState& state) {
      const size_t firstPending = applied_;

      // Fast-path: return cached evaluation directly if no moves are pending and king positions are unchanged.
      if (firstPending == history_.size()) {
        const Square whiteKing = NnueLayout::kingSquare(state, White);
        const Square blackKing = NnueLayout::kingSquare(state, Black);
        if (model_->hasAccumulator(White) && model_->hasAccumulator(Black) &&
            whiteKing == cachedKingSquares_[White] &&
            blackKing == cachedKingSquares_[Black]) {
          return model_->evaluate(state.getSide());
        }
      }

      // Update the accumulator for each perspective (White and Black).
      for (Side perspective = White; perspective <= Black; ++perspective) {
        const Square king = NnueLayout::kingSquare(state, perspective);

        // Rebuild accumulator from scratch if it is uninitialized or if the king has moved.
        if (!model_->hasAccumulator(perspective) || king != cachedKingSquares_[perspective]) {
          rebuild(state, perspective, king);
          continue;
        }

        const uint32_t kingBase = NnueLayout::kingBase(perspective, king);
        const size_t pendingCount = history_.size() - firstPending;

        // Fast-path for quiet moves: apply piece removal and addition in a single memory pass.
        if (pendingCount == 2 && !history_[firstPending].add && history_[firstPending + 1].add) {
          const auto& removedChange = history_[firstPending];
          const auto& addedChange = history_[firstPending + 1];
          const uint32_t subIndex = kingBase + NnueLayout::pieceOffset(
            perspective, removedChange.side, removedChange.piece, removedChange.square);
          const uint32_t addIndex = kingBase + NnueLayout::pieceOffset(
            perspective, addedChange.side, addedChange.piece, addedChange.square);
          model_->feedIncrementalPair(subIndex, addIndex, perspective);
        } else {
          // Fallback path: apply all pending piece additions and subtractions sequentially.
          for (size_t i = firstPending; i < history_.size(); ++i) {
            const PieceChange& change = history_[i];
            const uint32_t index = kingBase + NnueLayout::pieceOffset(
              perspective, change.side, change.piece, change.square);
            if (change.add) {
              model_->feedIncremental<true>(index, perspective);
            } else {
              model_->feedIncremental<false>(index, perspective);
            }
          }
        }
      }

      // Mark all pending changes as processed and compute the final evaluation score.
      applied_ = history_.size();
      return model_->evaluate(state.getSide());
    }
  };

  static_assert(EvaluationPolicy<NNUEEvaluationPolicy>);
}
