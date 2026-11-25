#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "nnue_mlp.h"
#include "nnue_serializer.h"
#include "searching_policy.h"
#include "small_vec.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace bb::evaluation {
  struct NnueLayout {
    static constexpr uint32_t kKingBucketCount = 16;
    static constexpr uint32_t kPieceTypes = 6; // Pawn, Knight, Bishop, Rook, Queen, King
    static constexpr uint32_t kKingPlaneStride = 2 * kPieceTypes * kSquareSize; // 768 inputs per king bucket
    static constexpr uint32_t kPiecePlaneStride = kSquareSize;                  // 64 squares per piece plane
    static constexpr uint32_t kInputSize = nnue::NnueModel::kInputSize;         // 12,288 total sparse inputs
    static constexpr uint32_t kFileMask = 7;
    static constexpr uint32_t kFoldCutoffFile = 3;
    static constexpr uint32_t kHorizontalFlipMask = 7;
    static constexpr uint32_t kRankShift = 3;
    static constexpr uint32_t kFoldedFiles = 4;
    using FeatureList = SmallVec<uint32_t, kSquareSize>;

    // Maps 32 horizontally-folded king squares (rank 8..1 top-to-bottom, files A..D) into 16 king buckets:
    //   Rank 1 (back rank): 4 individual buckets (0..3)
    //   Rank 2:             4 individual buckets (4..7)
    //   Rank 3:             2 file-pair buckets  (8..9)
    //   Rank 4:             2 file-pair buckets  (10..11)
    //   Ranks 5-6:          2 2x2 block buckets  (12..13)
    //   Ranks 7-8:          2 2x2 block buckets  (14..15)
    static constexpr std::array<uint8_t, 32> kKingBuckets{
      14, 14, 15, 15, // Rank 8 (foldedRank 0)
      14, 14, 15, 15, // Rank 7 (foldedRank 1)
      12, 12, 13, 13, // Rank 6 (foldedRank 2)
      12, 12, 13, 13, // Rank 5 (foldedRank 3)
      10, 10, 11, 11, // Rank 4 (foldedRank 4)
       8,  8,  9,  9, // Rank 3 (foldedRank 5)
       4,  5,  6,  7, // Rank 2 (foldedRank 6)
       0,  1,  2,  3, // Rank 1 (foldedRank 7)
    };

    static constexpr Square orient(Side perspective, Square square) noexcept {
      if (perspective == White) {
        return square;
      }
      return static_cast<Square>(square ^ kRankFlipMask);
    }

    static constexpr uint32_t kingFlipMask(Side perspective, Square kingSquare) noexcept {
      const Square orientedKing = orient(perspective, kingSquare);
      return ((orientedKing & kFileMask) > kFoldCutoffFile) ? kHorizontalFlipMask : 0;
    }

    static constexpr uint32_t kingIndex(Side perspective, Square kingSquare) noexcept {
      const Square orientedKing = orient(perspective, kingSquare);
      const uint32_t flipMask = ((orientedKing & kFileMask) > kFoldCutoffFile) ? kHorizontalFlipMask : 0;
      const Square foldedKing = static_cast<Square>(orientedKing ^ flipMask);
      const uint32_t foldedIdx = ((foldedKing >> kRankShift) * kFoldedFiles) + (foldedKing & kFileMask);
      return kKingBuckets[foldedIdx];
    }

    static constexpr uint32_t kingBase(Side perspective, Square kingSquare) noexcept {
      return kingIndex(perspective, kingSquare) * kKingPlaneStride;
    }

    static constexpr uint32_t pieceOffset(Side perspective, Side pieceSide, Piece piece, Square pieceSquare, uint32_t flipMask = 0) noexcept {
      const uint32_t relativeSide = pieceSide ^ perspective;
      const Square orientedPiece = orient(perspective, pieceSquare);
      const Square flippedPiece = static_cast<Square>(orientedPiece ^ flipMask);
      const uint32_t pieceIdx = relativeSide * kPieceTypes + piece;
      return pieceIdx * kPiecePlaneStride + flippedPiece;
    }

    static constexpr uint32_t pieceIndex(Side perspective, Side pieceSide, Piece piece, Square kingSquare, Square pieceSquare) noexcept {
      const uint32_t flipMask = kingFlipMask(perspective, kingSquare);
      return kingBase(perspective, kingSquare) + pieceOffset(perspective, pieceSide, piece, pieceSquare, flipMask);
    }

    static Square kingSquare(const BoardState& state, Side perspective) noexcept {
      return peekPiece(state.getPieces(perspective, King));
    }

    static uint32_t outputBucket(const BoardState& state) noexcept {
      const uint32_t totalPieces = countPiece(state.getOccupancy(White) | state.getOccupancy(Black));
      const uint32_t adjusted = totalPieces >= 2u ? (totalPieces - 2u) : 0u;
      return std::min(static_cast<uint32_t>(nnue::NnueModel::kOutputBuckets - 1), adjusted >> 2);
    }

    static FeatureList encode(const BoardState& state, Side perspective, Square kingSquare) {
      FeatureList features;
      const uint32_t flipMask = kingFlipMask(perspective, kingSquare);
      const uint32_t kBase = kingBase(perspective, kingSquare);
      for (Side relativeSide = White; relativeSide <= Black; ++relativeSide) {
        const Side pieceSide = perspective ^ relativeSide;
        for (Piece piece = Pawn; piece <= King; ++piece) {
          for (Bitboard pieces = state.getPieces(pieceSide, piece);
               pieces != 0;
               pieces = popPiece(pieces)) {
            features.push(kBase + pieceOffset(perspective, pieceSide, piece, peekPiece(pieces), flipMask));
          }
        }
      }
      return features;
    }
  };


  class NNUEEvaluationPolicy {
    static constexpr size_t kHistoryCapacity = 4 * searching::kMaxDepthHardCutoff;
    static constexpr size_t kStackCapacity = searching::kMaxDepthHardCutoff + 2;
    static constexpr size_t kAccumulatorSize = nnue::NnueModel::kAccumulatorSize;
    static constexpr size_t kQuietMovePieceChanges = 2;   // 1 removal + 1 placement
    static constexpr size_t kCaptureMovePieceChanges = 3; // 2 removals + 1 placement

    struct PieceChange {
      Side side;
      Piece piece;
      Square square;
      bool add;
    };

    struct alignas(32) StackEntry {
      alignas(32) std::array<std::array<int16_t, kAccumulatorSize>, 2> acc{};
      size_t historyPos = 0;
      int32_t cachedEval = 0;
      std::array<uint8_t, 2> kingBuckets{0, 0};
      std::array<uint8_t, 2> flipMasks{0, 0};
      uint8_t cachedStm = 2;
    };

    struct alignas(32) FinnyEntry {
      alignas(32) std::array<int16_t, kAccumulatorSize> acc{};
      std::array<std::array<Bitboard, NnueLayout::kPieceTypes>, 2> pieces{};
      bool initialized = false;
    };

    struct RuntimeState {
      std::array<StackEntry, kStackCapacity> stack{};
      std::array<std::array<std::array<FinnyEntry, 2>, NnueLayout::kKingBucketCount>, 2> finny{};
    };

    std::unique_ptr<nnue::NnueModel> model_;
    std::unique_ptr<RuntimeState> runtime_;
    SmallVec<PieceChange, kHistoryCapacity> history_;
    size_t stackSize_ = 0;

    void refreshFromFinny(const BoardState& state, Side perspective, uint8_t kingBucket, uint8_t flipMask,
                          uint32_t kingBase, int16_t* dst) noexcept {
      const size_t flipIdx = (flipMask != 0) ? 1u : 0u;
      FinnyEntry& entry = runtime_->finny[perspective][kingBucket][flipIdx];
      const auto& ft = model_->featureTransformer();

      if (!entry.initialized) [[unlikely]] {
        SmallVec<uint32_t, kSquareSize> features;
        for (Side side = White; side <= Black; ++side) {
          for (Piece piece = Pawn; piece <= King; ++piece) {
            const Bitboard bb = state.getPieces(side, piece);
            entry.pieces[side][piece] = bb;
            for (Bitboard bits = bb; bits != 0; bits = popPiece(bits)) {
              features.push(kingBase + NnueLayout::pieceOffset(perspective, side, piece, peekPiece(bits), flipMask));
            }
          }
        }
        ft.feedFull({features.begin(), features.size()}, entry.acc.data());
        entry.initialized = true;
      } else {
        SmallVec<uint32_t, kSquareSize> subs;
        SmallVec<uint32_t, kSquareSize> adds;
        for (Side side = White; side <= Black; ++side) {
          for (Piece piece = Pawn; piece <= King; ++piece) {
            const Bitboard oldBB = entry.pieces[side][piece];
            const Bitboard newBB = state.getPieces(side, piece);
            if (oldBB == newBB) {
              continue;
            }
            entry.pieces[side][piece] = newBB;
            for (Bitboard removed = oldBB & ~newBB; removed != 0; removed = popPiece(removed)) {
              subs.push(kingBase + NnueLayout::pieceOffset(perspective, side, piece, peekPiece(removed), flipMask));
            }
            for (Bitboard added = newBB & ~oldBB; added != 0; added = popPiece(added)) {
              adds.push(kingBase + NnueLayout::pieceOffset(perspective, side, piece, peekPiece(added), flipMask));
            }
          }
        }

        if (!subs.empty() || !adds.empty()) {
          ft.applyUpdates(entry.acc.data(), entry.acc.data(),
                          {subs.begin(), subs.size()}, {adds.begin(), adds.size()});
        }
      }

      ft.copyAccumulator(entry.acc.data(), dst);
    }

  public:
    explicit NNUEEvaluationPolicy(const std::filesystem::path& filename)
      : model_(nnue::NnueSerializer::load(filename)),
        runtime_(std::make_unique<RuntimeState>()) {}

    void reset() noexcept {
      history_.clear();
      stackSize_ = 0;
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      if (stackSize_ == 0) {
        return;
      }

      if (!history_.empty()) {
        const PieceChange& top = history_.back();
        if (top.side == side && top.piece == piece && top.square == square && top.add != Add) {
          history_.pop();
          while (stackSize_ > 0 && history_.size() < runtime_->stack[stackSize_ - 1].historyPos) {
            --stackSize_;
          }
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
      const Side stm = state.getSide();
      const uint32_t bucket = NnueLayout::outputBucket(state);

      // Fast-path: return cached evaluation directly if no pending piece changes exist.
      if (stackSize_ > 0 && runtime_->stack[stackSize_ - 1].historyPos == history_.size()) {
        StackEntry& top = runtime_->stack[stackSize_ - 1];
        if (top.cachedStm == static_cast<uint8_t>(stm)) {
          return top.cachedEval;
        }
        const int32_t score = model_->evaluate(top.acc[stm].data(), top.acc[stm ^ 1].data(), bucket);
        top.cachedEval = score;
        top.cachedStm = static_cast<uint8_t>(stm);
        return score;
      }

      const bool hasPrev = (stackSize_ > 0);
      const StackEntry* prev = hasPrev ? &runtime_->stack[stackSize_ - 1] : nullptr;
      const size_t firstPending = hasPrev ? prev->historyPos : 0;
      const size_t pendingCount = history_.size() - firstPending;

      // Push a new stack frame; if at capacity (e.g. long non-unmade sequence), update the top frame in place.
      StackEntry& cur = (stackSize_ < kStackCapacity)
        ? runtime_->stack[stackSize_++]
        : runtime_->stack[stackSize_ - 1];

      const auto& ft = model_->featureTransformer();

      for (Side perspective = White; perspective <= Black; ++perspective) {
        const Square kingSq = NnueLayout::kingSquare(state, perspective);
        const uint8_t kingBucket = static_cast<uint8_t>(NnueLayout::kingIndex(perspective, kingSq));
        const uint8_t flipMask = static_cast<uint8_t>(NnueLayout::kingFlipMask(perspective, kingSq));
        const uint32_t kingBase = static_cast<uint32_t>(kingBucket) * NnueLayout::kKingPlaneStride;

        const bool canIncremental = hasPrev
          && prev->kingBuckets[perspective] == kingBucket
          && prev->flipMasks[perspective] == flipMask;

        cur.kingBuckets[perspective] = kingBucket;
        cur.flipMasks[perspective] = flipMask;

        if (!canIncremental) [[unlikely]] {
          refreshFromFinny(state, perspective, kingBucket, flipMask, kingBase, cur.acc[perspective].data());
          continue;
        }

        const int16_t* srcAcc = prev->acc[perspective].data();
        int16_t* dstAcc = cur.acc[perspective].data();

        // Fast-path 1 for quiet moves: fused copy + 1 sub + 1 add in a single memory pass.
        if (pendingCount == kQuietMovePieceChanges && !history_[firstPending].add && history_[firstPending + 1].add) {
          const auto& removedChange = history_[firstPending];
          const auto& addedChange = history_[firstPending + 1];
          const uint32_t subIndex = kingBase + NnueLayout::pieceOffset(
            perspective, removedChange.side, removedChange.piece, removedChange.square, flipMask);
          const uint32_t addIndex = kingBase + NnueLayout::pieceOffset(
            perspective, addedChange.side, addedChange.piece, addedChange.square, flipMask);
          ft.copyAndUpdatePair(srcAcc, dstAcc, subIndex, addIndex);
        } else if (pendingCount == kCaptureMovePieceChanges &&
                   !history_[firstPending].add &&
                   !history_[firstPending + 1].add &&
                   history_[firstPending + 2].add) {
          // Fast-path 2 for captures: fused copy + 2 subs + 1 add in a single memory pass.
          const auto& sub0 = history_[firstPending];
          const auto& sub1 = history_[firstPending + 1];
          const auto& add0 = history_[firstPending + 2];
          const uint32_t subIndex0 = kingBase + NnueLayout::pieceOffset(
            perspective, sub0.side, sub0.piece, sub0.square, flipMask);
          const uint32_t subIndex1 = kingBase + NnueLayout::pieceOffset(
            perspective, sub1.side, sub1.piece, sub1.square, flipMask);
          const uint32_t addIndex = kingBase + NnueLayout::pieceOffset(
            perspective, add0.side, add0.piece, add0.square, flipMask);
          ft.copyAndUpdateCapture(srcAcc, dstAcc, subIndex0, subIndex1, addIndex);
        } else {
          SmallVec<uint32_t, kHistoryCapacity> subs;
          SmallVec<uint32_t, kHistoryCapacity> adds;
          for (size_t i = firstPending; i < history_.size(); ++i) {
            const PieceChange& change = history_[i];
            const uint32_t index = kingBase + NnueLayout::pieceOffset(
              perspective, change.side, change.piece, change.square, flipMask);
            if (change.add) {
              adds.push(index);
            } else {
              subs.push(index);
            }
          }
          ft.applyUpdates(srcAcc, dstAcc, {subs.begin(), subs.size()}, {adds.begin(), adds.size()});
        }
      }

      cur.historyPos = history_.size();
      const int32_t score = model_->evaluate(cur.acc[stm].data(), cur.acc[stm ^ 1].data(), bucket);
      cur.cachedEval = score;
      cur.cachedStm = static_cast<uint8_t>(stm);
      return score;
    }
  };

  static_assert(EvaluationPolicy<NNUEEvaluationPolicy>);
}
