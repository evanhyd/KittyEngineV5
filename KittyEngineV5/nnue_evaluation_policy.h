#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "nnue_mlp.h"
#include "nnue_serializer.h"
#include "searching_policy.h"
#include "small_vec.h"
#include <algorithm>
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
    static constexpr uint32_t kInputSize = nnue::NnueModel::kInputSize;         // 24,576 total sparse inputs
    static constexpr uint32_t kFileMask = 7;
    static constexpr uint32_t kFoldCutoffFile = 3;                              // Files D/E boundary (files E-H fold to A-D)
    static constexpr uint32_t kHorizontalFlipMask = 7;                          // XOR mask to fold files E-H to A-D
    static constexpr uint32_t kRankShift = 3;
    static constexpr uint32_t kFoldedFiles = 4;                                 // 4 files per rank in folded layout
    using FeatureList = SmallVec<uint32_t, kSquareSize>;

    static constexpr Square orient(Side perspective, Square square) noexcept {
      if (perspective == White) {
        return square;
      }
      return static_cast<Square>(square ^ kRankFlipMask);
    }

    static constexpr uint32_t getFlipMask(Square orientedKingSquare) noexcept {
      return ((orientedKingSquare & kFileMask) > kFoldCutoffFile) ? kHorizontalFlipMask : 0;
    }

    static constexpr uint32_t kingIndex(Square orientedKingSquare) noexcept {
      const uint32_t flipMask = getFlipMask(orientedKingSquare);
      const Square foldedKing = static_cast<Square>(orientedKingSquare ^ flipMask);
      return ((foldedKing >> kRankShift) * kFoldedFiles) + (foldedKing & kFileMask);
    }

    static constexpr uint32_t kingBase(Square orientedKingSquare) noexcept {
      return kingIndex(orientedKingSquare) * kKingPlaneStride;
    }

    static constexpr uint32_t pieceOffset(Side perspective, Side pieceSide, Piece piece, Square pieceSquare, uint32_t flipMask) noexcept {
      const uint32_t relativeSide = pieceSide ^ perspective;
      const Square orientedPiece = orient(perspective, pieceSquare);
      const Square foldedPiece = static_cast<Square>(orientedPiece ^ flipMask);
      const uint32_t pieceIdx = relativeSide * kPieceTypes + piece;
      return pieceIdx * kPiecePlaneStride + foldedPiece;
    }

    // Input Index: [folded king square: 32] * 768 + [piece index: 12] * 64 + [folded piece square: 64]
    static constexpr uint32_t pieceIndex(Side perspective, Side pieceSide, Piece piece, Square kingSquare, Square pieceSquare) noexcept {
      const Square orientedKing = orient(perspective, kingSquare);
      const uint32_t flip = getFlipMask(orientedKing);
      return kingBase(orientedKing) + pieceOffset(perspective, pieceSide, piece, pieceSquare, flip);
    }

    static Square kingSquare(const BoardState& state, Side perspective) {
      return peekPiece(state.getPieces(perspective, King));
    }

    static FeatureList encode(const BoardState& state, Side perspective,
                              uint32_t kBase, uint32_t flip) {
      FeatureList features;
      for (Side relativeSide = White; relativeSide <= Black; ++relativeSide) {
        const Side pieceSide = perspective ^ relativeSide;
        for (Piece piece = Pawn; piece <= King; ++piece) {
          for (Bitboard pieces = state.getPieces(pieceSide, piece);
               pieces != 0;
               pieces = popPiece(pieces)) {
            features.push(kBase + pieceOffset(perspective, pieceSide, piece, peekPiece(pieces), flip));
          }
        }
      }
      return features;
    }

    static FeatureList encode(const BoardState& state, Side perspective,
                              Square kingSquare) {
      const Square orientedKing = orient(perspective, kingSquare);
      return encode(state, perspective, kingBase(orientedKing), getFlipMask(orientedKing));
    }
  };


  class NNUEEvaluationPolicy {
    static constexpr size_t kHistoryCapacity = 4 * searching::kMaxDepthHardCutoff;
    static constexpr size_t kQuietMovePieceChanges = 2; // 1 removal + 1 placement
    static constexpr size_t kCaptureMovePieceChanges = 3; // 2 removals + 1 placement

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
    std::array<uint32_t, 2> cachedKingBases_{0, 0};
    std::array<uint32_t, 2> cachedFlipMasks_{0, 0};

    template <bool Add>
    void applyChange(Side perspective, Side side, Piece piece, Square square) noexcept {
      const uint32_t index = cachedKingBases_[perspective] + NnueLayout::pieceOffset(
        perspective, side, piece, square, cachedFlipMasks_[perspective]);
      model_->feedIncremental<Add>(index, perspective);
    }

    void rebuild(const BoardState& state, Side perspective, Square kingSquare) {
      cachedKingSquares_[perspective] = kingSquare;
      const Square orientedKing = NnueLayout::orient(perspective, kingSquare);
      const uint32_t kBase = NnueLayout::kingBase(orientedKing);
      const uint32_t flip = NnueLayout::getFlipMask(orientedKing);
      cachedKingBases_[perspective] = kBase;
      cachedFlipMasks_[perspective] = flip;

      const auto features = NnueLayout::encode(state, perspective, kBase, flip);
      model_->feedFull({features.begin(), features.size()}, perspective);
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
      // Calculate material piece count bucket (8 buckets from 2 to 32 pieces)
      const uint32_t totalPieces = countPiece(state.getOccupancy(White) | state.getOccupancy(Black));
      const uint32_t bucket = totalPieces <= 2 ? 0 : std::min(7u, (totalPieces - 2u) >> 2);

      const size_t firstPending = applied_;

      // Fast-path: return cached evaluation directly if no moves are pending and king positions are unchanged.
      if (firstPending == history_.size()) {
        const Square whiteKing = NnueLayout::kingSquare(state, White);
        const Square blackKing = NnueLayout::kingSquare(state, Black);
        if (model_->hasAccumulator(White) && model_->hasAccumulator(Black) &&
            whiteKing == cachedKingSquares_[White] &&
            blackKing == cachedKingSquares_[Black]) {
          return model_->evaluate(state.getSide(), bucket);
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

        const uint32_t kingBase = cachedKingBases_[perspective];
        const uint32_t flipMask = cachedFlipMasks_[perspective];
        const size_t pendingCount = history_.size() - firstPending;

        // Fast-path 1 for quiet moves: apply piece removal and addition in a single memory pass.
        if (pendingCount == kQuietMovePieceChanges && !history_[firstPending].add && history_[firstPending + 1].add) {
          const auto& removedChange = history_[firstPending];
          const auto& addedChange = history_[firstPending + 1];
          const uint32_t subIndex = kingBase + NnueLayout::pieceOffset(
            perspective, removedChange.side, removedChange.piece, removedChange.square, flipMask);
          const uint32_t addIndex = kingBase + NnueLayout::pieceOffset(
            perspective, addedChange.side, addedChange.piece, addedChange.square, flipMask);
          model_->feedIncrementalPair(subIndex, addIndex, perspective);
        } else if (pendingCount == kCaptureMovePieceChanges &&
                   !history_[firstPending].add &&
                   !history_[firstPending + 1].add &&
                   history_[firstPending + 2].add) {
          // Fast-path 2 for captures: 2 subs + 1 add in a single memory pass!
          const auto& sub0 = history_[firstPending];
          const auto& sub1 = history_[firstPending + 1];
          const auto& add0 = history_[firstPending + 2];
          const uint32_t subIndex0 = kingBase + NnueLayout::pieceOffset(
            perspective, sub0.side, sub0.piece, sub0.square, flipMask);
          const uint32_t subIndex1 = kingBase + NnueLayout::pieceOffset(
            perspective, sub1.side, sub1.piece, sub1.square, flipMask);
          const uint32_t addIndex = kingBase + NnueLayout::pieceOffset(
            perspective, add0.side, add0.piece, add0.square, flipMask);
          model_->feedIncrementalCapture(subIndex0, subIndex1, addIndex, perspective);
        } else {
          // Fallback path: apply all pending piece additions and subtractions sequentially.
          for (size_t i = firstPending; i < history_.size(); ++i) {
            const PieceChange& change = history_[i];
            const uint32_t index = kingBase + NnueLayout::pieceOffset(
              perspective, change.side, change.piece, change.square, flipMask);
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
      return model_->evaluate(state.getSide(), bucket);
    }
  };

  static_assert(EvaluationPolicy<NNUEEvaluationPolicy>);
}
