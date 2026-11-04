#pragma once
#include "nnue_matrix.h"
#include "nnue_math.h"
#include <array>
#include <span>
#include <algorithm>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace nnue {
  namespace internal {
    inline constexpr size_t kInt16PerBlock = sizeof(__m256i) / sizeof(int16_t); // 16
    inline constexpr size_t kInt8PerBlock  = sizeof(__m256i) / sizeof(int8_t);  // 32

    template <typename T>
    [[nodiscard]] inline __m256i loadVec(const T* ptr) noexcept {
      return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    template <typename T>
    inline void storeVec(T* ptr, __m256i vec) noexcept {
      _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), vec);
    }

    template <typename T>
    [[nodiscard]] inline __m128i loadVec128U(const T* ptr) noexcept {
      return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    }

    template <size_t neuronNum, size_t previousNeuronNum>
    struct Shape {
      static constexpr size_t kSize = (neuronNum + kInt16PerBlock - 1) / kInt16PerBlock * kInt16PerBlock;
      static constexpr size_t kBlockSize = kSize / kInt16PerBlock;
      static constexpr size_t kInputSize = (previousNeuronNum + kInt8PerBlock - 1) / kInt8PerBlock * kInt8PerBlock;
      static constexpr size_t kInputBlockSize = kInputSize / kInt8PerBlock;
    };
  }

  // Dual-perspective accumulator layer storing int16_t weights and biases for pure AVX2 integer updates.
  template <size_t neuronNum, size_t previousNeuronNum>
  class AccumulatorLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;
    static_assert(Shape::kSize == neuronNum, "AccumulatorLayer requires neuronNum to be a multiple of 16");
    static_assert(Shape::kBlockSize % 4 == 0, "AccumulatorLayer requires block size to be a multiple of 4 for unrolling");

    alignas(32) std::array<int16_t, Shape::kSize> biases_{};
    alignas(32) Matrix<int16_t, previousNeuronNum, Shape::kSize> weights_{};
    alignas(32) std::array<std::array<int16_t, Shape::kSize>, 2> accumulators_{};
    alignas(32) std::array<std::array<uint8_t, Shape::kSize>, 2> output_{};
    std::array<bool, 2> isInitialized{};
    std::array<bool, 2> isComputed{};

  public:
    AccumulatorLayer() = default;

    explicit AccumulatorLayer(std::span<const int16_t, neuronNum> biases,
                              const Matrix<int16_t, previousNeuronNum, neuronNum>& weights) noexcept {
      std::copy(biases.begin(), biases.end(), biases_.begin());
      for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
          weights_(i, j) = weights(i, j);
        }
      }
    }

    std::span<int16_t, neuronNum> biases() noexcept {
      return std::span<int16_t, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<int16_t, previousNeuronNum * neuronNum> weights() noexcept {
      static_assert(Shape::kSize == neuronNum);
      return std::span<int16_t, previousNeuronNum * neuronNum>(
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

      // Copy biases into accumulators with 2x unrolled AVX2 256-bit stores.
      for (size_t i = 0; i < Shape::kBlockSize; i += 2) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        internal::storeVec(&accumulators_[perspectiveIndex][offset0], internal::loadVec(&biases_[offset0]));
        internal::storeVec(&accumulators_[perspectiveIndex][offset1], internal::loadVec(&biases_[offset1]));
      }

      // Add active weight rows using 2x unrolled 16-bit integer SIMD additions.
      for (size_t inputIndex : activeIndex) {
        for (size_t i = 0; i < Shape::kBlockSize; i += 2) {
          const size_t offset0 = i * internal::kInt16PerBlock;
          const size_t offset1 = (i + 1) * internal::kInt16PerBlock;

          __m256i acc0 = internal::loadVec(&accumulators_[perspectiveIndex][offset0]);
          __m256i acc1 = internal::loadVec(&accumulators_[perspectiveIndex][offset1]);

          __m256i w0 = internal::loadVec(&weights_(inputIndex, offset0));
          __m256i w1 = internal::loadVec(&weights_(inputIndex, offset1));

          internal::storeVec(&accumulators_[perspectiveIndex][offset0], _mm256_add_epi16(acc0, w0));
          internal::storeVec(&accumulators_[perspectiveIndex][offset1], _mm256_add_epi16(acc1, w1));
        }
      }
    }

    template <bool isAdd>
    void feedIncremental(uint32_t inputIndex, uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = false;

      // 4x unrolled incremental updates for maximum instruction throughput and prefetching.
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

        __m256i acc0 = internal::loadVec(&accumulators_[perspectiveIndex][offset0]);
        __m256i acc1 = internal::loadVec(&accumulators_[perspectiveIndex][offset1]);
        __m256i acc2 = internal::loadVec(&accumulators_[perspectiveIndex][offset2]);
        __m256i acc3 = internal::loadVec(&accumulators_[perspectiveIndex][offset3]);

        __m256i w0 = internal::loadVec(&weights_(inputIndex, offset0));
        __m256i w1 = internal::loadVec(&weights_(inputIndex, offset1));
        __m256i w2 = internal::loadVec(&weights_(inputIndex, offset2));
        __m256i w3 = internal::loadVec(&weights_(inputIndex, offset3));

        if constexpr (isAdd) {
          acc0 = _mm256_add_epi16(acc0, w0);
          acc1 = _mm256_add_epi16(acc1, w1);
          acc2 = _mm256_add_epi16(acc2, w2);
          acc3 = _mm256_add_epi16(acc3, w3);
        } else {
          acc0 = _mm256_sub_epi16(acc0, w0);
          acc1 = _mm256_sub_epi16(acc1, w1);
          acc2 = _mm256_sub_epi16(acc2, w2);
          acc3 = _mm256_sub_epi16(acc3, w3);
        }

        internal::storeVec(&accumulators_[perspectiveIndex][offset0], acc0);
        internal::storeVec(&accumulators_[perspectiveIndex][offset1], acc1);
        internal::storeVec(&accumulators_[perspectiveIndex][offset2], acc2);
        internal::storeVec(&accumulators_[perspectiveIndex][offset3], acc3);
      }
    }

    std::array<std::span<const uint8_t, Shape::kSize>, 2> output() noexcept {
      const __m256i zero = _mm256_setzero_si256();
      const __m256i clamp_max = _mm256_set1_epi16(64);
      const __m256i clamp_max_act = _mm256_set1_epi16(std::numeric_limits<int8_t>::max());

      for (size_t p = 0; p < accumulators_.size(); ++p) {
        if (!isComputed[p]) {
          isComputed[p] = true;

          // Compute SCReLU: clamp to [0, 64], square (<= 4096), right-shift by 5, pack into uint8_t.
          for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
            const size_t offset0 = i * internal::kInt16PerBlock;
            const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
            const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
            const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

            __m256i a0 = internal::loadVec(&accumulators_[p][offset0]);
            __m256i a1 = internal::loadVec(&accumulators_[p][offset1]);
            __m256i a2 = internal::loadVec(&accumulators_[p][offset2]);
            __m256i a3 = internal::loadVec(&accumulators_[p][offset3]);

            a0 = _mm256_min_epi16(_mm256_max_epi16(a0, zero), clamp_max);
            a1 = _mm256_min_epi16(_mm256_max_epi16(a1, zero), clamp_max);
            a2 = _mm256_min_epi16(_mm256_max_epi16(a2, zero), clamp_max);
            a3 = _mm256_min_epi16(_mm256_max_epi16(a3, zero), clamp_max);

            __m256i sq0 = _mm256_mullo_epi16(a0, a0);
            __m256i sq1 = _mm256_mullo_epi16(a1, a1);
            __m256i sq2 = _mm256_mullo_epi16(a2, a2);
            __m256i sq3 = _mm256_mullo_epi16(a3, a3);

            __m256i sh0 = _mm256_min_epi16(_mm256_srai_epi16(sq0, 5), clamp_max_act);
            __m256i sh1 = _mm256_min_epi16(_mm256_srai_epi16(sq1, 5), clamp_max_act);
            __m256i sh2 = _mm256_min_epi16(_mm256_srai_epi16(sq2, 5), clamp_max_act);
            __m256i sh3 = _mm256_min_epi16(_mm256_srai_epi16(sq3, 5), clamp_max_act);

            __m256i packed0 = _mm256_packus_epi16(sh0, sh1);
            packed0 = _mm256_permute4x64_epi64(packed0, _MM_SHUFFLE(3, 1, 2, 0));

            __m256i packed1 = _mm256_packus_epi16(sh2, sh3);
            packed1 = _mm256_permute4x64_epi64(packed1, _MM_SHUFFLE(3, 1, 2, 0));

            internal::storeVec(&output_[p][offset0], packed0);
            internal::storeVec(&output_[p][offset2], packed1);
          }
        }
      }

      return { output_[0], output_[1] };
    }
  };

  // Dense Layer using int8_t weights, int32_t biases, and uint8_t activations via VPMADDUBSW.
  template <size_t neuronNum, size_t previousNeuronNum>
  class DenseLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;

    alignas(32) std::array<int32_t, Shape::kSize> biases_{};
    alignas(32) Matrix<int8_t, Shape::kSize, Shape::kInputSize> weights_{};
    alignas(32) std::array<uint8_t, Shape::kSize> output_{};

  public:
    DenseLayer() = default;

    explicit DenseLayer(std::span<const int32_t, neuronNum> biases,
                        const Matrix<int8_t, neuronNum, previousNeuronNum>& weights) noexcept {
      std::copy(biases.begin(), biases.end(), biases_.begin());
      for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
          weights_(i, j) = weights(i, j);
        }
      }
    }

    std::span<int32_t, neuronNum> biases() noexcept {
      return std::span<int32_t, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<int8_t, neuronNum * previousNeuronNum> weights() noexcept {
      static_assert(Shape::kSize == neuronNum && Shape::kInputSize == previousNeuronNum);
      return std::span<int8_t, neuronNum * previousNeuronNum>(
        weights_.data(), neuronNum * previousNeuronNum);
    }

    void feedFull(std::span<const uint8_t, Shape::kInputSize> inputs) noexcept {
      const __m256i ones = _mm256_set1_epi16(1);
      const __m128i zero128 = _mm_setzero_si128();
      const __m128i clamp8192 = _mm_set1_epi32(8192);
      const __m128i clamp_max_act = _mm_set1_epi32(std::numeric_limits<int8_t>::max());

      size_t row = 0;
      // Process 4 neurons concurrently with SIMD reduction and SIMD activation.
      for (; row + 4 <= neuronNum; row += 4) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        size_t col = 0;
        // 2-block unrolled inner loop (64 bytes per iteration)
        for (; col + 2 * internal::kInt8PerBlock <= Shape::kInputSize; col += 2 * internal::kInt8PerBlock) {
          const size_t col0 = col;
          const size_t col1 = col + internal::kInt8PerBlock;

          __m256i in0 = internal::loadVec(&inputs[col0]);
          __m256i in1 = internal::loadVec(&inputs[col1]);

          __m256i w0_0 = internal::loadVec(&weights_(row + 0, col0));
          __m256i w0_1 = internal::loadVec(&weights_(row + 0, col1));
          __m256i w1_0 = internal::loadVec(&weights_(row + 1, col0));
          __m256i w1_1 = internal::loadVec(&weights_(row + 1, col1));
          __m256i w2_0 = internal::loadVec(&weights_(row + 2, col0));
          __m256i w2_1 = internal::loadVec(&weights_(row + 2, col1));
          __m256i w3_0 = internal::loadVec(&weights_(row + 3, col0));
          __m256i w3_1 = internal::loadVec(&weights_(row + 3, col1));

          __m256i p0_0 = _mm256_maddubs_epi16(in0, w0_0);
          __m256i p0_1 = _mm256_maddubs_epi16(in1, w0_1);
          __m256i p1_0 = _mm256_maddubs_epi16(in0, w1_0);
          __m256i p1_1 = _mm256_maddubs_epi16(in1, w1_1);
          __m256i p2_0 = _mm256_maddubs_epi16(in0, w2_0);
          __m256i p2_1 = _mm256_maddubs_epi16(in1, w2_1);
          __m256i p3_0 = _mm256_maddubs_epi16(in0, w3_0);
          __m256i p3_1 = _mm256_maddubs_epi16(in1, w3_1);

          sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0_0, ones));
          sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0_1, ones));

          sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1_0, ones));
          sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1_1, ones));

          sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2_0, ones));
          sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2_1, ones));

          sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3_0, ones));
          sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3_1, ones));
        }

        for (; col < Shape::kInputSize; col += internal::kInt8PerBlock) {
          __m256i in = internal::loadVec(&inputs[col]);

          __m256i w0 = internal::loadVec(&weights_(row + 0, col));
          __m256i w1 = internal::loadVec(&weights_(row + 1, col));
          __m256i w2 = internal::loadVec(&weights_(row + 2, col));
          __m256i w3 = internal::loadVec(&weights_(row + 3, col));

          __m256i p0 = _mm256_maddubs_epi16(in, w0);
          __m256i p1 = _mm256_maddubs_epi16(in, w1);
          __m256i p2 = _mm256_maddubs_epi16(in, w2);
          __m256i p3 = _mm256_maddubs_epi16(in, w3);

          sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(p0, ones));
          sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(p1, ones));
          sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(p2, ones));
          sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(p3, ones));
        }

        // 4-way parallel horizontal reduction of sum0..sum3
        __m256i s01 = _mm256_hadd_epi32(sum0, sum1);
        __m256i s23 = _mm256_hadd_epi32(sum2, sum3);
        __m256i h2 = _mm256_hadd_epi32(s01, s23);
        __m128i dots = _mm_add_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));

        // Vectorized bias add and SCReLU activation
        __m128i b4 = internal::loadVec128U(&biases_[row]);
        __m128i z4 = _mm_add_epi32(dots, b4);
        __m128i c4 = _mm_min_epi32(_mm_max_epi32(z4, zero128), clamp8192);
        __m128i v4 = _mm_srli_epi32(c4, 6);
        __m128i sq4 = _mm_mullo_epi32(v4, v4);
        __m128i act4 = _mm_min_epi32(_mm_srli_epi32(sq4, 7), clamp_max_act);

        // Pack 4 32-bit values into 4 bytes and store to output
        __m128i pack16 = _mm_packus_epi32(act4, act4);
        __m128i pack8 = _mm_packus_epi16(pack16, pack16);
        const uint32_t val4 = static_cast<uint32_t>(_mm_cvtsi128_si32(pack8));
        *reinterpret_cast<uint32_t*>(&output_[row]) = val4;
      }

      for (; row < neuronNum; ++row) {
        __m256i sum = _mm256_setzero_si256();
        for (size_t col = 0; col < Shape::kInputSize; col += internal::kInt8PerBlock) {
          __m256i in = internal::loadVec(&inputs[col]);
          __m256i w = internal::loadVec(&weights_(row, col));
          __m256i p = _mm256_maddubs_epi16(in, w);
          sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, ones));
        }
        const int32_t dot = horizontalAdd32(sum);
        const int32_t z = dot + biases_[row];
        const int32_t c = std::clamp(z, 0, 8192);
        const int32_t v = c >> 6;
        output_[row] = static_cast<uint8_t>(std::min<int32_t>((v * v) >> 7, std::numeric_limits<int8_t>::max()));
      }
    }

    std::span<const uint8_t, Shape::kSize> output() const noexcept {
      return output_;
    }
  };

  // Single-neuron output layer computing final centipawn score.
  template <size_t previousNeuronNum, auto activation = nnue::scaledSigmoid>
  class OutputLayer {
    using Shape = internal::Shape<1, previousNeuronNum>;

    alignas(32) std::array<int16_t, Shape::kInputSize> weights_{};
    int32_t bias_{};
    float output_{};

  public:
    OutputLayer() = default;

    explicit OutputLayer(std::span<const int32_t, 1> biases,
                         const Matrix<int16_t, 1, previousNeuronNum>& weights) noexcept
      : bias_(biases[0]) {
      for (size_t j = 0; j < previousNeuronNum; ++j) {
        weights_[j] = weights(0, j);
      }
    }

    std::span<int32_t, 1> biases() noexcept {
      return std::span<int32_t, 1>(&bias_, 1);
    }

    std::span<int16_t, previousNeuronNum> weights() noexcept {
      static_assert(Shape::kInputSize == previousNeuronNum);
      return std::span<int16_t, previousNeuronNum>(weights_.data(), previousNeuronNum);
    }

    void feedFull(std::span<const uint8_t, Shape::kInputSize> inputs) noexcept {
      static_assert(Shape::kInputSize % 32 == 0, "OutputLayer requires input size to be a multiple of 32");

      __m256i sumVec = _mm256_setzero_si256();
      for (size_t col = 0; col < Shape::kInputSize; col += 32) {
        // Zero-extend 32 unsigned 8-bit activations into two 16-element signed 16-bit vectors.
        __m128i in128_low = internal::loadVec128U(inputs.data() + col);
        __m128i in128_high = internal::loadVec128U(inputs.data() + col + 16);

        __m256i in16_low = _mm256_cvtepu8_epi16(in128_low);
        __m256i in16_high = _mm256_cvtepu8_epi16(in128_high);

        __m256i w16_low = internal::loadVec(weights_.data() + col);
        __m256i w16_high = internal::loadVec(weights_.data() + col + 16);

        __m256i prod_low = _mm256_madd_epi16(in16_low, w16_low);
        __m256i prod_high = _mm256_madd_epi16(in16_high, w16_high);

        sumVec = _mm256_add_epi32(sumVec, _mm256_add_epi32(prod_low, prod_high));
      }
      const int32_t total = horizontalAdd32(sumVec) + bias_;

      // Descale by 2048 (2^11) directly to centipawns with nearest-integer rounding (+1024).
      const int32_t eval_cp = (total + 1024) >> 11;
      output_ = activation(static_cast<float>(eval_cp));
    }

    float output() const noexcept {
      return output_;
    }
  };
}
