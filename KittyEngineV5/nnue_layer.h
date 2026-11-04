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
      return _mm256_load_si256(static_cast<const __m256i*>(static_cast<const void*>(ptr)));
    }

    template <typename T>
    inline void storeVec(T* ptr, __m256i vec) noexcept {
      _mm256_store_si256(static_cast<__m256i*>(static_cast<void*>(ptr)), vec);
    }

    template <typename T>
    [[nodiscard]] inline __m128i loadVec128U(const T* ptr) noexcept {
      return _mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(ptr)));
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

    void feedIncrementalPair(uint32_t subIndex, uint32_t addIndex, uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = false;

      // 4x unrolled single-pass sub + add (saves 33% memory bandwidth on quiet moves).
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

        __m256i weightSub0 = internal::loadVec(&weights_(subIndex, offset0));
        __m256i weightSub1 = internal::loadVec(&weights_(subIndex, offset1));
        __m256i weightSub2 = internal::loadVec(&weights_(subIndex, offset2));
        __m256i weightSub3 = internal::loadVec(&weights_(subIndex, offset3));

        __m256i weightAdd0 = internal::loadVec(&weights_(addIndex, offset0));
        __m256i weightAdd1 = internal::loadVec(&weights_(addIndex, offset1));
        __m256i weightAdd2 = internal::loadVec(&weights_(addIndex, offset2));
        __m256i weightAdd3 = internal::loadVec(&weights_(addIndex, offset3));

        __m256i diff0 = _mm256_sub_epi16(weightAdd0, weightSub0);
        __m256i diff1 = _mm256_sub_epi16(weightAdd1, weightSub1);
        __m256i diff2 = _mm256_sub_epi16(weightAdd2, weightSub2);
        __m256i diff3 = _mm256_sub_epi16(weightAdd3, weightSub3);

        __m256i acc0 = internal::loadVec(&accumulators_[perspectiveIndex][offset0]);
        __m256i acc1 = internal::loadVec(&accumulators_[perspectiveIndex][offset1]);
        __m256i acc2 = internal::loadVec(&accumulators_[perspectiveIndex][offset2]);
        __m256i acc3 = internal::loadVec(&accumulators_[perspectiveIndex][offset3]);

        acc0 = _mm256_add_epi16(acc0, diff0);
        acc1 = _mm256_add_epi16(acc1, diff1);
        acc2 = _mm256_add_epi16(acc2, diff2);
        acc3 = _mm256_add_epi16(acc3, diff3);

        internal::storeVec(&accumulators_[perspectiveIndex][offset0], acc0);
        internal::storeVec(&accumulators_[perspectiveIndex][offset1], acc1);
        internal::storeVec(&accumulators_[perspectiveIndex][offset2], acc2);
        internal::storeVec(&accumulators_[perspectiveIndex][offset3], acc3);
      }
    }

    std::array<std::span<const uint8_t, Shape::kSize>, 2> output() noexcept {
      const __m256i zero = _mm256_setzero_si256();
      const __m256i clampMax = _mm256_set1_epi16(64);
      const __m256i clampMaxActivation = _mm256_set1_epi16(std::numeric_limits<int8_t>::max());

      for (size_t p = 0; p < accumulators_.size(); ++p) {
        if (!isComputed[p]) {
          isComputed[p] = true;

          // Compute SCReLU: clamp to [0, 64], square (<= 4096), right-shift by 5, pack into uint8_t.
          for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
            const size_t offset0 = i * internal::kInt16PerBlock;
            const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
            const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
            const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

            __m256i acc0 = internal::loadVec(&accumulators_[p][offset0]);
            __m256i acc1 = internal::loadVec(&accumulators_[p][offset1]);
            __m256i acc2 = internal::loadVec(&accumulators_[p][offset2]);
            __m256i acc3 = internal::loadVec(&accumulators_[p][offset3]);

            acc0 = _mm256_min_epi16(_mm256_max_epi16(acc0, zero), clampMax);
            acc1 = _mm256_min_epi16(_mm256_max_epi16(acc1, zero), clampMax);
            acc2 = _mm256_min_epi16(_mm256_max_epi16(acc2, zero), clampMax);
            acc3 = _mm256_min_epi16(_mm256_max_epi16(acc3, zero), clampMax);

            __m256i sq0 = _mm256_mullo_epi16(acc0, acc0);
            __m256i sq1 = _mm256_mullo_epi16(acc1, acc1);
            __m256i sq2 = _mm256_mullo_epi16(acc2, acc2);
            __m256i sq3 = _mm256_mullo_epi16(acc3, acc3);

            __m256i shift0 = _mm256_min_epi16(_mm256_srai_epi16(sq0, 5), clampMaxActivation);
            __m256i shift1 = _mm256_min_epi16(_mm256_srai_epi16(sq1, 5), clampMaxActivation);
            __m256i shift2 = _mm256_min_epi16(_mm256_srai_epi16(sq2, 5), clampMaxActivation);
            __m256i shift3 = _mm256_min_epi16(_mm256_srai_epi16(sq3, 5), clampMaxActivation);

            __m256i packed0 = _mm256_packus_epi16(shift0, shift1);
            packed0 = _mm256_permute4x64_epi64(packed0, _MM_SHUFFLE(3, 1, 2, 0));

            __m256i packed1 = _mm256_packus_epi16(shift2, shift3);
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

    // Vectorized 8-way reduction and activation pipeline into output_[row..row+7].
    inline void reduceAndActivate8(size_t row,
                                   __m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3,
                                   __m256i sum4, __m256i sum5, __m256i sum6, __m256i sum7) noexcept {
      const __m256i s01 = _mm256_hadd_epi32(sum0, sum1);
      const __m256i s23 = _mm256_hadd_epi32(sum2, sum3);
      const __m256i s45 = _mm256_hadd_epi32(sum4, sum5);
      const __m256i s67 = _mm256_hadd_epi32(sum6, sum7);

      const __m256i h0123 = _mm256_hadd_epi32(s01, s23);
      const __m256i h4567 = _mm256_hadd_epi32(s45, s67);

      const __m128i dots0123 = _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
      const __m128i dots4567 = _mm_add_epi32(_mm256_castsi256_si128(h4567), _mm256_extracti128_si256(h4567, 1));

      const __m256i dots8 = _mm256_set_m128i(dots4567, dots0123);
      const __m256i biases8 = internal::loadVec(&biases_[row]);
      const __m256i sumWithBiases = _mm256_add_epi32(dots8, biases8);
      const __m256i clamped = _mm256_min_epi32(_mm256_max_epi32(sumWithBiases, _mm256_setzero_si256()), _mm256_set1_epi32(8192));
      const __m256i scaled = _mm256_srli_epi32(clamped, 6);
      const __m256i squared = _mm256_mullo_epi32(scaled, scaled);
      const __m256i activated = _mm256_min_epi32(_mm256_srli_epi32(squared, 7), _mm256_set1_epi32(std::numeric_limits<int8_t>::max()));

      const __m128i activatedLow = _mm256_castsi256_si128(activated);
      const __m128i activatedHigh = _mm256_extracti128_si256(activated, 1);
      const __m128i pack16 = _mm_packus_epi32(activatedLow, activatedHigh);
      const __m128i pack8 = _mm_packus_epi16(pack16, pack16);
      const uint64_t val8 = static_cast<uint64_t>(_mm_cvtsi128_si64(pack8));
      std::memcpy(&output_[row], &val8, sizeof(val8));
    }

    // Vectorized 4-way reduction and activation pipeline into output_[row..row+3].
    inline void reduceAndActivate4(size_t row,
                                   __m256i sum0, __m256i sum1, __m256i sum2, __m256i sum3) noexcept {
      const __m256i s01 = _mm256_hadd_epi32(sum0, sum1);
      const __m256i s23 = _mm256_hadd_epi32(sum2, sum3);
      const __m256i h2 = _mm256_hadd_epi32(s01, s23);
      const __m128i dots = _mm_add_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));

      const __m128i biases4 = internal::loadVec128U(&biases_[row]);
      const __m128i sumWithBiases = _mm_add_epi32(dots, biases4);
      const __m128i clamped = _mm_min_epi32(_mm_max_epi32(sumWithBiases, _mm_setzero_si128()), _mm_set1_epi32(8192));
      const __m128i scaled = _mm_srli_epi32(clamped, 6);
      const __m128i squared = _mm_mullo_epi32(scaled, scaled);
      const __m128i activated = _mm_min_epi32(_mm_srli_epi32(squared, 7), _mm_set1_epi32(std::numeric_limits<int8_t>::max()));

      const __m128i pack16 = _mm_packus_epi32(activated, activated);
      const __m128i pack8 = _mm_packus_epi16(pack16, pack16);
      const uint32_t val4 = static_cast<uint32_t>(_mm_cvtsi128_si32(pack8));
      std::memcpy(&output_[row], &val4, sizeof(val4));
    }

    // Scalar reduction and activation fallback.
    inline void reduceAndActivateScalar(size_t row, __m256i sum) noexcept {
      const int32_t dotProduct = horizontalAdd32(sum);
      const int32_t sumWithBias = dotProduct + biases_[row];
      const int32_t clamped = std::clamp(sumWithBias, 0, 8192);
      const int32_t scaled = clamped >> 6;
      output_[row] = static_cast<uint8_t>(std::min<int32_t>((scaled * scaled) >> 7, std::numeric_limits<int8_t>::max()));
    }

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

      // Fast-path specialization for 32-input layers (e.g. secondHidden): load input once.
      if constexpr (Shape::kInputSize == 32) {
        const __m256i in = internal::loadVec(inputs.data());

        size_t row = 0;
        for (; row + 8 <= neuronNum; row += 8) {
          __m256i p0 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 0, 0)));
          __m256i p1 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 1, 0)));
          __m256i p2 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 2, 0)));
          __m256i p3 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 3, 0)));
          __m256i p4 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 4, 0)));
          __m256i p5 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 5, 0)));
          __m256i p6 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 6, 0)));
          __m256i p7 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 7, 0)));

          __m256i sum0 = _mm256_madd_epi16(p0, ones);
          __m256i sum1 = _mm256_madd_epi16(p1, ones);
          __m256i sum2 = _mm256_madd_epi16(p2, ones);
          __m256i sum3 = _mm256_madd_epi16(p3, ones);
          __m256i sum4 = _mm256_madd_epi16(p4, ones);
          __m256i sum5 = _mm256_madd_epi16(p5, ones);
          __m256i sum6 = _mm256_madd_epi16(p6, ones);
          __m256i sum7 = _mm256_madd_epi16(p7, ones);

          reduceAndActivate8(row, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
        }
        for (; row + 4 <= neuronNum; row += 4) {
          __m256i p0 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 0, 0)));
          __m256i p1 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 1, 0)));
          __m256i p2 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 2, 0)));
          __m256i p3 = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 3, 0)));

          __m256i sum0 = _mm256_madd_epi16(p0, ones);
          __m256i sum1 = _mm256_madd_epi16(p1, ones);
          __m256i sum2 = _mm256_madd_epi16(p2, ones);
          __m256i sum3 = _mm256_madd_epi16(p3, ones);

          reduceAndActivate4(row, sum0, sum1, sum2, sum3);
        }
        for (; row < neuronNum; ++row) {
          __m256i p = _mm256_maddubs_epi16(in, internal::loadVec(&weights_(row, 0)));
          __m256i sum = _mm256_madd_epi16(p, ones);
          reduceAndActivateScalar(row, sum);
        }
        return;
      }

      // General case: 8-way concurrent neuron evaluation
      size_t row = 0;
      for (; row + 8 <= neuronNum; row += 8) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        __m256i sum4 = _mm256_setzero_si256();
        __m256i sum5 = _mm256_setzero_si256();
        __m256i sum6 = _mm256_setzero_si256();
        __m256i sum7 = _mm256_setzero_si256();

        for (size_t col = 0; col < Shape::kInputSize; col += internal::kInt8PerBlock) {
          __m256i in = internal::loadVec(&inputs[col]);

          sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 0, col))), ones));
          sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 1, col))), ones));
          sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 2, col))), ones));
          sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 3, col))), ones));
          sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 4, col))), ones));
          sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 5, col))), ones));
          sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 6, col))), ones));
          sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 7, col))), ones));
        }

        reduceAndActivate8(row, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
      }
      for (; row + 4 <= neuronNum; row += 4) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();

        for (size_t col = 0; col < Shape::kInputSize; col += internal::kInt8PerBlock) {
          __m256i in = internal::loadVec(&inputs[col]);

          sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 0, col))), ones));
          sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 1, col))), ones));
          sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 2, col))), ones));
          sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 3, col))), ones));
        }

        reduceAndActivate4(row, sum0, sum1, sum2, sum3);
      }
      for (; row < neuronNum; ++row) {
        __m256i sum = _mm256_setzero_si256();
        for (size_t col = 0; col < Shape::kInputSize; col += internal::kInt8PerBlock) {
          __m256i in = internal::loadVec(&inputs[col]);
          __m256i w = internal::loadVec(&weights_(row, col));
          __m256i p = _mm256_maddubs_epi16(in, w);
          sum = _mm256_add_epi32(sum, _mm256_madd_epi16(p, ones));
        }
        reduceAndActivateScalar(row, sum);
      }
    }

    // Zero-copy dual-span feedFull for two contiguous halves (e.g. White + Black accumulators).
    template <size_t HalfSize = Shape::kInputSize / 2>
    requires (HalfSize * 2 == Shape::kInputSize)
    void feedFull(std::span<const uint8_t, HalfSize> in0,
                  std::span<const uint8_t, HalfSize> in1) noexcept {
      const __m256i ones = _mm256_set1_epi16(1);

      size_t row = 0;
      for (; row + 8 <= neuronNum; row += 8) {
        __m256i sum0 = _mm256_setzero_si256();
        __m256i sum1 = _mm256_setzero_si256();
        __m256i sum2 = _mm256_setzero_si256();
        __m256i sum3 = _mm256_setzero_si256();
        __m256i sum4 = _mm256_setzero_si256();
        __m256i sum5 = _mm256_setzero_si256();
        __m256i sum6 = _mm256_setzero_si256();
        __m256i sum7 = _mm256_setzero_si256();

        auto accumulateBlock = [&](std::span<const uint8_t, HalfSize> inputSpan, size_t columnOffset) {
          for (size_t col = 0; col < HalfSize; col += internal::kInt8PerBlock) {
            const size_t weightColumn = columnOffset + col;
            __m256i in = internal::loadVec(&inputSpan[col]);

            sum0 = _mm256_add_epi32(sum0, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 0, weightColumn))), ones));
            sum1 = _mm256_add_epi32(sum1, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 1, weightColumn))), ones));
            sum2 = _mm256_add_epi32(sum2, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 2, weightColumn))), ones));
            sum3 = _mm256_add_epi32(sum3, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 3, weightColumn))), ones));
            sum4 = _mm256_add_epi32(sum4, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 4, weightColumn))), ones));
            sum5 = _mm256_add_epi32(sum5, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 5, weightColumn))), ones));
            sum6 = _mm256_add_epi32(sum6, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 6, weightColumn))), ones));
            sum7 = _mm256_add_epi32(sum7, _mm256_madd_epi16(_mm256_maddubs_epi16(in, internal::loadVec(&weights_(row + 7, weightColumn))), ones));
          }
        };

        accumulateBlock(in0, 0);
        accumulateBlock(in1, HalfSize);

        reduceAndActivate8(row, sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
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
    int32_t output_{};

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
        __m128i inputLow128 = internal::loadVec128U(inputs.data() + col);
        __m128i inputHigh128 = internal::loadVec128U(inputs.data() + col + 16);

        __m256i inputLow16 = _mm256_cvtepu8_epi16(inputLow128);
        __m256i inputHigh16 = _mm256_cvtepu8_epi16(inputHigh128);

        __m256i weightLow = internal::loadVec(weights_.data() + col);
        __m256i weightHigh = internal::loadVec(weights_.data() + col + 16);

        __m256i productLow = _mm256_madd_epi16(inputLow16, weightLow);
        __m256i productHigh = _mm256_madd_epi16(inputHigh16, weightHigh);

        sumVec = _mm256_add_epi32(sumVec, _mm256_add_epi32(productLow, productHigh));
      }
      const int32_t total = horizontalAdd32(sumVec) + bias_;

      // Descale by 2048 (2^11) directly to centipawns with nearest-integer rounding (+1024).
      output_ = (total + 1024) >> 11;
    }

    int32_t output() const noexcept {
      return output_;
    }
  };
}
