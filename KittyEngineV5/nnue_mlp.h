#pragma once
#include "nnue_layer.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace nnue {
  class NnueModel {
  public:
    static constexpr size_t kInputSize = 24'576;
    static constexpr size_t kAccumulatorSize = 256;
    static constexpr size_t kHidden1Size = 32;
    static constexpr size_t kHidden2Size = 32;
    static constexpr size_t kOutputBuckets = 8;

  private:
    AccumulatorLayer<kAccumulatorSize, kInputSize> accumulator_;
    DenseLayer<kHidden1Size, 2 * kAccumulatorSize> firstHidden_;
    DenseLayer<kHidden2Size, kHidden1Size> secondHidden_;
    OutputLayer<kHidden2Size, kOutputBuckets> output_;

  public:
    AccumulatorLayer<kAccumulatorSize, kInputSize>& accumulator() noexcept {
      return accumulator_;
    }

    DenseLayer<kHidden1Size, 2 * kAccumulatorSize>& firstHidden() noexcept {
      return firstHidden_;
    }

    DenseLayer<kHidden2Size, kHidden1Size>& secondHidden() noexcept {
      return secondHidden_;
    }

    OutputLayer<kHidden2Size, kOutputBuckets>& outputLayer() noexcept {
      return output_;
    }

    void clearAccumulators() noexcept {
      accumulator_.reset();
    }

    bool hasAccumulator(uint32_t perspectiveIndex) const noexcept {
      return accumulator_.initialized(perspectiveIndex);
    }

    void feedFull(std::span<const uint32_t> features, uint32_t perspectiveIndex) {
      accumulator_.feedFull(features, perspectiveIndex);
    }

    template <bool Add>
    void feedIncremental(uint32_t index, uint32_t perspectiveIndex) {
      accumulator_.template feedIncremental<Add>(index, perspectiveIndex);
    }

    void feedIncrementalPair(uint32_t subIndex, uint32_t addIndex, uint32_t perspectiveIndex) noexcept {
      accumulator_.feedIncrementalPair(subIndex, addIndex, perspectiveIndex);
    }

    void feedIncrementalCapture(uint32_t subIndex0, uint32_t subIndex1, uint32_t addIndex, uint32_t perspectiveIndex) noexcept {
      accumulator_.feedIncrementalCapture(subIndex0, subIndex1, addIndex, perspectiveIndex);
    }

    int32_t evaluate(uint32_t perspectiveIndex, uint32_t bucket) {
      // Retrieve activated accumulator outputs for both perspectives (side to move first, then opponent).
      const auto perspectives = accumulator_.output();
      const auto selected = perspectives[perspectiveIndex];
      const auto opposite = perspectives[perspectiveIndex ^ 1];

      // Forward activations through the first and second hidden layers.
      firstHidden_.feedFull(selected, opposite);
      secondHidden_.feedFull(firstHidden_.output());

      // Compute final output layer activation for the selected material/piece bucket and return centipawns.
      return output_.feedFull(secondHidden_.output(), bucket);
    }
  };
}
