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
    static constexpr uint32_t kPieceTypes = King; // 5 non-king piece types (Pawn, Knight, Bishop, Rook, Queen)
    static constexpr uint32_t kKingPlaneStride = 2 * kPieceTypes * kSquareSize; // 640 inputs per king square
    static constexpr uint32_t kPiecePlaneStride = kSquareSize;                  // 64 squares per piece plane
    static constexpr uint32_t kInputSize = nnue::NnueModel::kInputSize;         // 40,960 total sparse inputs
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

    // Input Index: [own king square: 64] * 640 + [piece index: 10] * 64 + [piece square: 64]
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
        for (Piece piece = Pawn; piece < King; ++piece) {
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
    // A capture with promotion changes at most three non-king features per ply.
    static constexpr size_t kHistoryCapacity = 3 * searching::kMaxDepthHardCutoff;

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
        return;
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
      for (Side perspective = White; perspective <= Black; ++perspective) {
        const Square king = NnueLayout::kingSquare(state, perspective);
        if (!model_->hasAccumulator(perspective) || king != cachedKingSquares_[perspective]) {
          rebuild(state, perspective, king);
          continue;
        }

        const uint32_t kingBase = NnueLayout::kingBase(perspective, king);
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

      applied_ = history_.size();
      return static_cast<int32_t>(std::lround(model_->evaluate(state.getSideToMove())));
    }
  };

  static_assert(EvaluationPolicy<NNUEEvaluationPolicy>);
}
