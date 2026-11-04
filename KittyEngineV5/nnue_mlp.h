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
    static constexpr size_t kInputSize = 49'152;
    static constexpr size_t kAccumulatorSize = 256;
    static constexpr size_t kHiddenSize = 32;

  private:
    AccumulatorLayer<kAccumulatorSize, kInputSize> accumulator_;
    DenseLayer<kHiddenSize, 2 * kAccumulatorSize> firstHidden_;
    DenseLayer<kHiddenSize, kHiddenSize> secondHidden_;
    OutputLayer<kHiddenSize, linear> output_;

  public:
    AccumulatorLayer<kAccumulatorSize, kInputSize>& accumulator() noexcept {
      return accumulator_;
    }

    DenseLayer<kHiddenSize, 2 * kAccumulatorSize>& firstHidden() noexcept {
      return firstHidden_;
    }

    DenseLayer<kHiddenSize, kHiddenSize>& secondHidden() noexcept {
      return secondHidden_;
    }

    OutputLayer<kHiddenSize, linear>& outputLayer() noexcept {
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

    int32_t evaluate(uint32_t perspectiveIndex) {
      // Retrieve activated accumulator outputs for both perspectives (side to move first, then opponent).
      const auto perspectives = accumulator_.output();
      const auto selected = perspectives[perspectiveIndex];
      const auto opposite = perspectives[perspectiveIndex ^ 1];

      // Forward activations through the first and second hidden layers.
      firstHidden_.feedFull(selected, opposite);
      secondHidden_.feedFull(firstHidden_.output());

      // Compute final output layer activation and return the centipawn score directly.
      output_.feedFull(secondHidden_.output());
      return output_.outputCp();
    }
  };
}
