#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "nnue_mlp.h"
#include "nnue_serializer.h"
#include "nnue_math.h"
#include "searching_policy.h"
#include "small_vec.h"
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

namespace bb::evaluation {
  struct NnueLayout {
    static constexpr uint32_t kPieceTypes = King; // 5 non-king piece types (Pawn, Knight, Bishop, Rook, Queen)
    static constexpr uint32_t kKingPlaneStride = 2 * kPieceTypes * kSquareSize; // 640 inputs per king square
    static constexpr uint32_t kPiecePlaneStride = kSquareSize;                  // 64 squares per piece plane
    static constexpr uint32_t kInputSize = nnue::NnueModel::kInputSize;         // 40,960 total sparse inputs
    using FeatureList = SmallVec<uint32_t, kSquareSize>;

    static constexpr Square orient(Side perspective, Square square) noexcept {
      return Square(perspective == White ? square : square ^ kRankFlipMask);
    }

    // Input Index: [own king square: 64] * 640 + [piece index: 10] * 64 + [piece square: 64]
    static constexpr uint32_t pieceIndex(Side perspective, Side pieceSide, Piece piece, Square kingSquare, Square pieceSquare) noexcept {
      const uint32_t relativeSide = pieceSide ^ perspective;
      const uint32_t orientedKing = orient(perspective, kingSquare);
      const uint32_t orientedPiece = orient(perspective, pieceSquare);
      const uint32_t pieceIdx = relativeSide * kPieceTypes + piece;
      return orientedKing * kKingPlaneStride + pieceIdx * kPiecePlaneStride + orientedPiece;
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


  // NNUE MLP MODEL // 
  class MLPEvaluationPolicy {
    struct PieceChange {
      Side side;
      Piece piece;
      Square square;
      bool add;
    };

    std::unique_ptr<nnue::NnueModel> model_;
    SmallVec<PieceChange, searching::kMaxDepthHardCutoff> pending_;
    std::array<Square, 2> cachedKingSquares_{NoSquare, NoSquare};
    std::optional<ZobristHash::Hash> preparedRootHash_;
    bool fullRefreshNeeded_ = false;

    void rebuild(const BoardState& state, Side perspective, Square kingSquare) {
      const auto features = NnueLayout::encode(state, perspective, kingSquare);
      model_->feedFull({features.begin(), features.size()}, perspective);
      cachedKingSquares_[perspective] = kingSquare;
    }

  public:
    explicit MLPEvaluationPolicy(const std::filesystem::path& filename)
      : model_(nnue::NnueSerializer::load(filename)) {}

    void reset() noexcept {
      pending_.clear();
      cachedKingSquares_.fill(NoSquare);
      preparedRootHash_.reset();
      fullRefreshNeeded_ = false;
      model_->clearAccumulators();
    }

    void prepare(const BoardState& state) {
      if (preparedRootHash_.has_value() && *preparedRootHash_ != state.getHash() && pending_.empty()) {
        reset();
      }
      preparedRootHash_ = state.getHash();
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      if (piece == King || fullRefreshNeeded_ || !model_->hasAccumulator(White)) {
        return;
      }

      // Keep only the net difference from the last evaluated position.
      for (size_t i = 0; i < pending_.size(); ++i) {
        const PieceChange& change = pending_[i];
        if (change.side != side || change.piece != piece || change.square != square) {
          continue;
        }
        pending_[i] = pending_.back();
        pending_.pop();
        return;
      }

      if (pending_.size() == searching::kMaxDepthHardCutoff) {
        fullRefreshNeeded_ = true;
        pending_.clear();
        return;
      }
      pending_.push({side, piece, square, Add});
    }

    int32_t evaluate(const BoardState& state) {
      const std::array<Square, 2> currentKings{
        NnueLayout::kingSquare(state, White),
        NnueLayout::kingSquare(state, Black)
      };

      for (Side perspective = White; perspective <= Black; ++perspective) {
        if (fullRefreshNeeded_ || !model_->hasAccumulator(perspective) || currentKings[perspective] != cachedKingSquares_[perspective]) {
          rebuild(state, perspective, currentKings[perspective]);
          continue;
        }

        // Change first-layer sums only for pieces that differ from the cached board.
        for (const PieceChange& change : pending_) {
          const uint32_t index = NnueLayout::pieceIndex(
            perspective, change.side, change.piece,
            currentKings[perspective], change.square);
          if (change.add) {
            model_->feedIncremental<true>(index, perspective);
          } else {
            model_->feedIncremental<false>(index, perspective);
          }
        }
      }

      pending_.clear();
      fullRefreshNeeded_ = false;
      return nnue::scaledSigmoidToCentipawns(model_->evaluate(state.getSideToMove()));
    }
  };

  static_assert(EvaluationPolicy<MLPEvaluationPolicy>);
}
