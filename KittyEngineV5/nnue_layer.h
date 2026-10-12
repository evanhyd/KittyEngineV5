#pragma once
#include "nnue_matrix.h"
#include "nnue_math.h"
#include <array>
#include <span>
#include <algorithm>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nnue {
  namespace internal {
    inline constexpr size_t kFloatPerBlock = sizeof(__m256) / sizeof(float);

    template <size_t neuronNum, size_t previousNeuronNum>
    struct Shape {
      static constexpr size_t kSize = (neuronNum + kFloatPerBlock - 1) / kFloatPerBlock * kFloatPerBlock;
      static constexpr size_t kBlockSize = kSize / kFloatPerBlock;
      static constexpr size_t kInputSize = (previousNeuronNum + kFloatPerBlock - 1) / kFloatPerBlock * kFloatPerBlock;
      static constexpr size_t kInputBlockSize = kInputSize / kFloatPerBlock;
    };

  }

  // Each accumulator caches the features for one perspective index.
  // Contains raw output before applying the activation function. Feed input must be binary.
  template <size_t neuronNum, size_t previousNeuronNum>
  class AccumulatorLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;

    std::array<bool, 2> isInitialized{};
    std::array<bool, 2> isComputed{};
    alignas(__m256) std::array<float, Shape::kSize> biases_{};
    alignas(__m256) Matrix<float, previousNeuronNum, Shape::kSize> weights_{}; // [previous][current]. Because input it sparse but weight must be contiguous.
    alignas(__m256) std::array<std::array<float, Shape::kSize>, 2> accumulators_{};
    alignas(__m256) std::array<std::array<float, Shape::kSize>, 2> output_{};

  public:
    AccumulatorLayer() = default;

    explicit AccumulatorLayer(std::span<const float, neuronNum> biases, const Matrix<float, previousNeuronNum, neuronNum>& weights) noexcept {
      std::copy(biases.begin(), biases.end(), biases_.begin());

      for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
          weights_(i, j) = weights(i, j);
        }
      }
    }

    std::span<float, neuronNum> biases() noexcept {
      return std::span<float, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<float, previousNeuronNum * neuronNum> weights() noexcept {
      static_assert(Shape::kSize == neuronNum);
      return std::span<float, previousNeuronNum * neuronNum>(
        weights_.data(), previousNeuronNum * neuronNum);
    }

    void reset() noexcept {
      isInitialized.fill(false);
      isComputed.fill(false);
    }

    bool initialized(uint32_t perspectiveIndex) const noexcept {
      return isInitialized[perspectiveIndex];
    }

    void feedFull(std::span<const uint32_t> activeIndex, uint32_t perspectiveIndex) {
      isInitialized[perspectiveIndex] = true;
      isComputed[perspectiveIndex] = false;

      // Add the biases.
      for (size_t i = 0; i < Shape::kBlockSize; ++i) {
        const size_t offset = i * internal::kFloatPerBlock;
        _mm256_store_ps(&accumulators_[perspectiveIndex][offset], _mm256_load_ps(&biases_[offset]));
      }

      // Add the weight to the accumulator row by row.
      for (size_t inputIndex : activeIndex) {
        for (size_t i = 0; i < Shape::kBlockSize; ++i) {
          const size_t offset = i * internal::kFloatPerBlock;
          // y += xW, where x is 1.
          __m256 accumulated = _mm256_load_ps(&accumulators_[perspectiveIndex][offset]);
          __m256 weight = _mm256_load_ps(&weights_(inputIndex, offset));
          _mm256_store_ps(&accumulators_[perspectiveIndex][offset], _mm256_add_ps(accumulated, weight));
        }
      }
    }

    // Inputs must be aligned to __m256.
    template <bool isAdd>
    void feedIncremental(uint32_t inputIndex, uint32_t perspectiveIndex) {
      isComputed[perspectiveIndex] = false;

      for (size_t i = 0; i < Shape::kBlockSize; ++i) {
        const size_t offset = i * internal::kFloatPerBlock;
        __m256 rawOutput = _mm256_load_ps(&accumulators_[perspectiveIndex][offset]);

        // y += xW, where x is 1.
        __m256 weight = _mm256_load_ps(&weights_(inputIndex, offset));

        if constexpr (isAdd) {
          rawOutput = _mm256_add_ps(rawOutput, weight);
        } else {
          rawOutput = _mm256_sub_ps(rawOutput, weight);
        }

        // Store raw output.
        _mm256_store_ps(&accumulators_[perspectiveIndex][offset], rawOutput);
      }
    }

    std::array<std::span<const float, Shape::kSize>, 2> output() {
      for (size_t p = 0; p < accumulators_.size(); ++p) {
        if (!isComputed[p]) {
          isComputed[p] = true;

          // Apply leaky ReLU and store in the final output.
          for (size_t i = 0; i < Shape::kBlockSize; ++i) {
            const size_t offset = i * internal::kFloatPerBlock;
            __m256 rawOutput = _mm256_load_ps(&accumulators_[p][offset]);
            _mm256_store_ps(&output_[p][offset], nnue::leakyRelu(rawOutput));
          }
        }
      }

      return { output_[0], output_[1]};
    }
  };

  // Dense Layer. Always does full forwad propagation.
  template <size_t neuronNum, size_t previousNeuronNum>
  class DenseLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;

    alignas(__m256) std::array<float, Shape::kSize> biases_{};
    alignas(__m256) Matrix<float, Shape::kSize, Shape::kInputSize> weights_{}; // [current][previous].
    alignas(__m256) std::array<float, Shape::kSize> output_{};

    template <size_t... i>
    void bulkLoadWeights(std::index_sequence<i...>,
                         std::array<__m256, internal::kFloatPerBlock>& weights, size_t iOffset, size_t jOffset) {
      ((weights[i] = _mm256_load_ps(&weights_(iOffset + i, jOffset))), ...);
    }

    template <size_t... i>
    void bulkFuseMultiplyAdd(std::index_sequence<i...>,
                             std::array<__m256, internal::kFloatPerBlock>& output, __m256 inputs, size_t iOffset, size_t jOffset) {
      ((output[i] = _mm256_fmadd_ps(inputs, _mm256_load_ps(&weights_(iOffset + i, jOffset)), output[i])), ...);
    }

    float reduceSum(__m256 v) {
      alignas(__m256) std::array<float, internal::kFloatPerBlock> vals;
      _mm256_store_ps(vals.data(), v);
      return ((vals[0] + vals[1]) + (vals[2] + vals[3])) + ((vals[4] + vals[5]) + (vals[6] + vals[7]));
    }

    template <size_t... i>
    __m256 bulkReduceSum(std::index_sequence<i...>,
                         const std::array<__m256, internal::kFloatPerBlock>& outputs) {
      return _mm256_setr_ps(reduceSum(outputs[i])...);
    }

  public:
    DenseLayer() = default;

    explicit DenseLayer(std::span<const float, neuronNum> biases, const Matrix<float, neuronNum, previousNeuronNum>& weights) noexcept {
      std::copy(biases.begin(), biases.end(), biases_.begin());

      for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
          weights_(i, j) = weights(i, j);
        }
      }
    }

    std::span<float, neuronNum> biases() noexcept {
      return std::span<float, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<float, neuronNum * previousNeuronNum> weights() noexcept {
      static_assert(Shape::kSize == neuronNum && Shape::kInputSize == previousNeuronNum);
      return std::span<float, neuronNum * previousNeuronNum>(
        weights_.data(), neuronNum * previousNeuronNum);
    }

    // Inputs must be aligned to __m256.
    void feedFull(std::span<const float, Shape::kInputSize> inputs) {
      for (size_t i = 0; i < Shape::kBlockSize; ++i) {
        const size_t iOffset = i * internal::kFloatPerBlock;

        std::array<__m256, internal::kFloatPerBlock> outputs;
        outputs.fill(_mm256_setzero_ps());

        // Calculate input * weight multiple rows at a time.
        // Multiple input per row at a time.
        for (size_t j = 0; j < Shape::kInputBlockSize; ++j) {
          const size_t jOffset = j * internal::kFloatPerBlock;
          __m256 input = _mm256_load_ps(&inputs[jOffset]);
          bulkFuseMultiplyAdd(std::make_index_sequence<internal::kFloatPerBlock>(), outputs, input, iOffset, jOffset);
        }

        // Calculate the dot product for each neuron, then add the biases.
        __m256 dotProduct = bulkReduceSum(std::make_index_sequence<internal::kFloatPerBlock>(), outputs);
        __m256 outputAfterReduce = _mm256_add_ps(dotProduct, _mm256_load_ps(&biases_[iOffset]));

        // Apply leaky ReLU.
        _mm256_store_ps(&output_[iOffset], nnue::leakyRelu(outputAfterReduce));
      }
    }


    std::span<const float, Shape::kSize> output() const {
      return output_;
    }
  };

  // Single-neuron dense layer; the activation is chosen at compile time.
  template <size_t previousNeuronNum, auto activation = nnue::scaledSigmoid>
  class OutputLayer {
    using Shape = internal::Shape<1, previousNeuronNum>;

    float bias_{};
    alignas(__m256) std::array<float, Shape::kInputSize> weights_{};
    float output_{};

  public:
    OutputLayer() = default;

    explicit OutputLayer(std::span<const float, 1> biases,
                         const Matrix<float, 1, previousNeuronNum>& weights) noexcept
      : bias_(biases[0]) {
      for (size_t j = 0; j < previousNeuronNum; ++j) {
        weights_[j] = weights(0, j);
      }
    }

    std::span<float, 1> biases() noexcept {
      return std::span<float, 1>(&bias_, 1);
    }

    std::span<float, previousNeuronNum> weights() noexcept {
      static_assert(Shape::kInputSize == previousNeuronNum);
      return std::span<float, previousNeuronNum>(weights_.data(), previousNeuronNum);
    }

    void feedFull(std::span<const float, Shape::kInputSize> inputs) {
      __m256 products = _mm256_setzero_ps();
      for (size_t j = 0; j < Shape::kInputSize; j += internal::kFloatPerBlock) {
        const __m256 input = _mm256_loadu_ps(inputs.data() + j);
        const __m256 weight = _mm256_load_ps(weights_.data() + j);
        products = _mm256_fmadd_ps(input, weight, products);
      }

      alignas(__m256) std::array<float, internal::kFloatPerBlock> values;
      _mm256_store_ps(values.data(), products);
      const float sum = ((values[0] + values[1]) + (values[2] + values[3])) +
                        ((values[4] + values[5]) + (values[6] + values[7]));
      output_ = activation(sum + bias_);
    }

    float output() const {
      return output_;
    }
  };
}
