#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "mlp_math.h"
#include "mlp_serializer.h"
#include "nnue.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <optional>

namespace bb::evaluation {
  class MLPEvaluationPolicy {
    static constexpr bb::Square kRankFlipMask = 56;
    static constexpr size_t kPieceInputSize = bb::kSideSize * bb::kPieceSize * bb::kSquareSize;
    static constexpr size_t kCastlingInputIndex = kPieceInputSize;
    static constexpr size_t kEnpassantInputIndex = kCastlingInputIndex + 4;
    static constexpr size_t kInputSize = kEnpassantInputIndex + bb::kSquareSize;
    static constexpr std::array<size_t, 4> kLayerSizes{kInputSize, 256, 32, 1};
    static constexpr std::array<std::array<CastlePermission, 4>, 2> kCastlingRights{{
      {WhiteKingCastle, WhiteQueenCastle, BlackKingCastle, BlackQueenCastle},
      {BlackKingCastle, BlackQueenCastle, WhiteKingCastle, WhiteQueenCastle}
    }};
    mlp::NNUE model_;
    std::optional<ZobristHash::Hash> rootHash_;

    static constexpr size_t pieceInputIndex(Side perspective, Side pieceSide, Piece piece, Square square) noexcept {
      const Square relativeSquare = square ^ (perspective == Black ? kRankFlipMask : 0);
      return ((pieceSide ^ perspective) * kPieceSize + piece) * kSquareSize + relativeSquare;
    }

  public:
    template <bb::Side Mover>
    static std::array<float, kInputSize> encodePosition(const BoardState& state) {
      std::array<float, kInputSize> input{};
      for (bb::Side relativeSide = 0; relativeSide < bb::kSideSize; ++relativeSide) {
        const bb::Side pieceSide = Mover ^ relativeSide;

        for (bb::Piece piece = bb::Pawn; piece < bb::kPieceSize; ++piece) {
          for (bb::Bitboard pieces = state.getPieces(pieceSide, piece); 
               pieces != 0; 
               pieces = bb::popPiece(pieces)) {
            const bb::Square square = bb::peekPiece(pieces);
            input[pieceInputIndex(Mover, pieceSide, piece, square)] = 1.0f;
          }
        }
      }

      const bb::CastlePermission castling = state.getCastlingRights();
      // Castling rights are ordered relative to this perspective.
      for (size_t i = 0; i < kCastlingRights[Mover].size(); ++i) {
        input[kCastlingInputIndex + i] = (castling & kCastlingRights[Mover][i]) != 0;
      }

      const bb::Square enPassant = state.getEnpassantSquare();
      if (enPassant != bb::NoSquare) {
        input[kEnpassantInputIndex + (enPassant ^ (Mover == Black ? kRankFlipMask : 0))] = 1.0f;
      }
      return input;
    }

  private:
    // Convert win-draw-lose space to centi-pawn space.
    static int32_t wdlToCp(float wdl) {
      wdl = std::clamp(wdl, 1e-6f, 1.0f - 1e-6f);
      return static_cast<int32_t>(std::lround(400.0f * std::log(wdl / (1.0f - wdl))));
    }

  public:
    explicit MLPEvaluationPolicy(const std::filesystem::path& filename)
      : model_(kLayerSizes, mlp::relu, mlp::scaledSigmoid) {
      mlp::MLPSerializer(model_).load(filename);
    }

    void reset() {
      rootHash_.reset();
      model_.clearAccumulators();
    }

    void prepare(const BoardState& state) {
      if (rootHash_ == state.getHash() && model_.hasAccumulator(White)) {
        return;
      }
      const auto whiteInput = encodePosition<White>(state);
      const auto blackInput = encodePosition<Black>(state);
      model_.initializeAccumulator(White, whiteInput);
      model_.initializeAccumulator(Black, blackInput);
      rootHash_ = state.getHash();
    }

    template <bool Add>
    void markPiece(Side side, Piece piece, Square square) noexcept {
      if (!model_.hasAccumulator(White)) {
        return;
      }
      for (Side perspective = White; perspective <= Black; ++perspective) {
        model_.setInput(perspective, pieceInputIndex(perspective, side, piece, square), Add ? 1.0f : 0.0f);
      }
    }

    void markCastle(CastlePermission changedRights) noexcept {
      if (!model_.hasAccumulator(White) || changedRights == 0) {
        return;
      }
      for (Side perspective = White; perspective <= Black; ++perspective) {
        for (size_t i = 0; i < 4; ++i) {
          if ((changedRights & kCastlingRights[perspective][i]) == 0) {
            continue;
          }
          const size_t index = kCastlingInputIndex + i;
          model_.setInput(perspective, index, 1.0f - model_.getInput(perspective, index));
        }
      }
    }

    void markEnpassant(Square square) noexcept {
      if (!model_.hasAccumulator(White) || square == NoSquare) {
        return;
      }
      for (Side perspective = White; perspective <= Black; ++perspective) {
        const size_t index = kEnpassantInputIndex + (square ^ (perspective == Black ? kRankFlipMask : 0));
        model_.setInput(perspective, index, 1.0f - model_.getInput(perspective, index));
      }
    }

    int32_t evaluate(const BoardState& state) {
      if (!model_.hasAccumulator(White)) {
        prepare(state);
      }
      return wdlToCp(model_.inferPerspective(state.getSideToMove())[0]);
    }
  };

  static_assert(EvaluationPolicy<MLPEvaluationPolicy>);
}
