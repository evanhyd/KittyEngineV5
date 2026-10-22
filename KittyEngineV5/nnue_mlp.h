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
    alignas(__m256) std::array<float, 2 * kAccumulatorSize> combined_{};

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

    float evaluate(uint32_t perspectiveIndex) {
      const auto perspectives = accumulator_.output();
      const auto selected = perspectives[perspectiveIndex];
      const auto opposite = perspectives[perspectiveIndex ^ 1];
      std::copy(selected.begin(), selected.end(), combined_.begin());
      std::copy(opposite.begin(), opposite.end(), combined_.begin() + kAccumulatorSize);
      firstHidden_.feedFull(combined_);
      secondHidden_.feedFull(firstHidden_.output());
      output_.feedFull(secondHidden_.output());
      return output_.output();
    }
  };
}
