#pragma once
#include "nnue_matrix.h"
#include "nnue_math.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <span>

namespace nnue {
  namespace internal {
    inline constexpr size_t kInt16PerBlock = sizeof(__m256i) / sizeof(int16_t); // 16
    inline constexpr size_t kInt8PerBlock  = sizeof(__m256i) / sizeof(int8_t);  // 32
    inline constexpr int32_t kActivationMax = std::numeric_limits<int8_t>::max(); // 127
    inline constexpr int16_t kAccumulatorActivationClamp = 64;
    inline constexpr int32_t kAccumulatorSquareShift = 5;
    inline constexpr int32_t kAccumulatorPermute = _MM_SHUFFLE(3, 1, 2, 0);
    inline constexpr int32_t kDenseActivationClamp = 8192;
    inline constexpr int32_t kDenseActivationScaleShift = 6;
    inline constexpr int32_t kDenseActivationSquareShift = 7;
    inline constexpr int32_t kOutputShift = 11;
    inline constexpr int32_t kOutputRounding = 1 << (kOutputShift - 1); // 1024

    template <typename T>
    [[nodiscard]] inline __m256i loadVec(const T* ptr) noexcept {
      return _mm256_load_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    template <typename T>
    inline void storeVec(T* ptr, __m256i vec) noexcept {
      _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), vec);
    }

    [[nodiscard]] inline __m256i activateSCReLU(__m256i sumWithBiases) noexcept {
      const __m256i zero = _mm256_setzero_si256();
      const __m256i clampMax = _mm256_set1_epi32(kDenseActivationClamp);
      const __m256i maxAct = _mm256_set1_epi32(kActivationMax);

      const __m256i clamped = _mm256_min_epi32(_mm256_max_epi32(sumWithBiases, zero), clampMax);
      const __m256i scaled = _mm256_srli_epi32(clamped, kDenseActivationScaleShift);
      const __m256i squared = _mm256_mullo_epi32(scaled, scaled);
      return _mm256_min_epi32(_mm256_srli_epi32(squared, kDenseActivationSquareShift), maxAct);
    }

    template <size_t neuronNum, size_t previousNeuronNum>
    struct Shape {
      static constexpr size_t kSize = (neuronNum + kInt16PerBlock - 1) / kInt16PerBlock * kInt16PerBlock;
      static constexpr size_t kBlockSize = kSize / kInt16PerBlock;
      static constexpr size_t kInputSize = (previousNeuronNum + kInt8PerBlock - 1) / kInt8PerBlock * kInt8PerBlock;
    };
  }

  // Dual-perspective accumulator layer storing int16_t weights and biases for pure AVX2 integer updates.
  template <size_t neuronNum, size_t previousNeuronNum>
  class AccumulatorLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;
    static_assert(Shape::kSize == neuronNum, "AccumulatorLayer requires neuronNum to be a multiple of 16");
    static_assert(Shape::kBlockSize % 4 == 0, "AccumulatorLayer requires block size to be a multiple of 4 for unrolling");

    static constexpr size_t kPerspectives = 2;

    alignas(32) std::array<int16_t, Shape::kSize> biases_{};
    alignas(32) Matrix<int16_t, previousNeuronNum, Shape::kSize> weights_{};
    alignas(32) std::array<std::array<int16_t, Shape::kSize>, kPerspectives> accumulators_{};
    alignas(32) std::array<std::array<uint8_t, Shape::kSize>, kPerspectives> output_{};
    std::array<bool, kPerspectives> isInitialized{};
    std::array<bool, kPerspectives> isComputed{};

    void computeOutput(uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = true;
      const int16_t* acc = accumulators_[perspectiveIndex].data();
      uint8_t* out = output_[perspectiveIndex].data();

      const __m256i zero = _mm256_setzero_si256();
      const __m256i clampMax = _mm256_set1_epi16(internal::kAccumulatorActivationClamp);
      const __m256i clampMaxActivation = _mm256_set1_epi16(static_cast<int16_t>(internal::kActivationMax));

      // Compute SCReLU: clamp to [0, 64], square (<= 4096), right-shift by 5, pack into uint8_t.
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

        __m256i acc0 = internal::loadVec(&acc[offset0]);
        __m256i acc1 = internal::loadVec(&acc[offset1]);
        __m256i acc2 = internal::loadVec(&acc[offset2]);
        __m256i acc3 = internal::loadVec(&acc[offset3]);

        acc0 = _mm256_min_epi16(_mm256_max_epi16(acc0, zero), clampMax);
        acc1 = _mm256_min_epi16(_mm256_max_epi16(acc1, zero), clampMax);
        acc2 = _mm256_min_epi16(_mm256_max_epi16(acc2, zero), clampMax);
        acc3 = _mm256_min_epi16(_mm256_max_epi16(acc3, zero), clampMax);

        __m256i sq0 = _mm256_mullo_epi16(acc0, acc0);
        __m256i sq1 = _mm256_mullo_epi16(acc1, acc1);
        __m256i sq2 = _mm256_mullo_epi16(acc2, acc2);
        __m256i sq3 = _mm256_mullo_epi16(acc3, acc3);

        __m256i shift0 = _mm256_min_epi16(_mm256_srai_epi16(sq0, internal::kAccumulatorSquareShift), clampMaxActivation);
        __m256i shift1 = _mm256_min_epi16(_mm256_srai_epi16(sq1, internal::kAccumulatorSquareShift), clampMaxActivation);
        __m256i shift2 = _mm256_min_epi16(_mm256_srai_epi16(sq2, internal::kAccumulatorSquareShift), clampMaxActivation);
        __m256i shift3 = _mm256_min_epi16(_mm256_srai_epi16(sq3, internal::kAccumulatorSquareShift), clampMaxActivation);

        __m256i packed0 = _mm256_packus_epi16(shift0, shift1);
        packed0 = _mm256_permute4x64_epi64(packed0, internal::kAccumulatorPermute);

        __m256i packed1 = _mm256_packus_epi16(shift2, shift3);
        packed1 = _mm256_permute4x64_epi64(packed1, internal::kAccumulatorPermute);

        internal::storeVec(&out[offset0], packed0);
        internal::storeVec(&out[offset2], packed1);
      }
    }

  public:
    AccumulatorLayer() = default;

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

    void feedFull(std::span<const uint32_t> activeIndex, uint32_t perspectiveIndex) noexcept {
      isInitialized[perspectiveIndex] = true;
      isComputed[perspectiveIndex] = false;
      int16_t* acc = accumulators_[perspectiveIndex].data();

      // Copy biases into accumulators with 4x unrolled AVX2 256-bit stores.
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;
        internal::storeVec(&acc[offset0], internal::loadVec(&biases_[offset0]));
        internal::storeVec(&acc[offset1], internal::loadVec(&biases_[offset1]));
        internal::storeVec(&acc[offset2], internal::loadVec(&biases_[offset2]));
        internal::storeVec(&acc[offset3], internal::loadVec(&biases_[offset3]));
      }

      // Add active weight rows using 4x unrolled 16-bit integer SIMD additions.
      for (size_t inputIndex : activeIndex) {
        for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
          const size_t offset0 = i * internal::kInt16PerBlock;
          const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
          const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
          const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

          __m256i acc0 = internal::loadVec(&acc[offset0]);
          __m256i acc1 = internal::loadVec(&acc[offset1]);
          __m256i acc2 = internal::loadVec(&acc[offset2]);
          __m256i acc3 = internal::loadVec(&acc[offset3]);

          __m256i w0 = internal::loadVec(&weights_(inputIndex, offset0));
          __m256i w1 = internal::loadVec(&weights_(inputIndex, offset1));
          __m256i w2 = internal::loadVec(&weights_(inputIndex, offset2));
          __m256i w3 = internal::loadVec(&weights_(inputIndex, offset3));

          acc0 = _mm256_add_epi16(acc0, w0);
          acc1 = _mm256_add_epi16(acc1, w1);
          acc2 = _mm256_add_epi16(acc2, w2);
          acc3 = _mm256_add_epi16(acc3, w3);

          internal::storeVec(&acc[offset0], acc0);
          internal::storeVec(&acc[offset1], acc1);
          internal::storeVec(&acc[offset2], acc2);
          internal::storeVec(&acc[offset3], acc3);
        }
      }
    }

    template <bool isAdd>
    void feedIncremental(uint32_t inputIndex, uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = false;
      int16_t* acc = accumulators_[perspectiveIndex].data();
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

        __m256i acc0 = internal::loadVec(&acc[offset0]);
        __m256i acc1 = internal::loadVec(&acc[offset1]);
        __m256i acc2 = internal::loadVec(&acc[offset2]);
        __m256i acc3 = internal::loadVec(&acc[offset3]);

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

        internal::storeVec(&acc[offset0], acc0);
        internal::storeVec(&acc[offset1], acc1);
        internal::storeVec(&acc[offset2], acc2);
        internal::storeVec(&acc[offset3], acc3);
      }
    }

    void feedIncrementalPair(uint32_t subIndex, uint32_t addIndex, uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = false;
      int16_t* acc = accumulators_[perspectiveIndex].data();
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

        __m256i acc0 = internal::loadVec(&acc[offset0]);
        __m256i acc1 = internal::loadVec(&acc[offset1]);
        __m256i acc2 = internal::loadVec(&acc[offset2]);
        __m256i acc3 = internal::loadVec(&acc[offset3]);

        acc0 = _mm256_add_epi16(acc0, diff0);
        acc1 = _mm256_add_epi16(acc1, diff1);
        acc2 = _mm256_add_epi16(acc2, diff2);
        acc3 = _mm256_add_epi16(acc3, diff3);

        internal::storeVec(&acc[offset0], acc0);
        internal::storeVec(&acc[offset1], acc1);
        internal::storeVec(&acc[offset2], acc2);
        internal::storeVec(&acc[offset3], acc3);
      }
    }

    void feedIncrementalCapture(uint32_t subIndex0, uint32_t subIndex1, uint32_t addIndex, uint32_t perspectiveIndex) noexcept {
      isComputed[perspectiveIndex] = false;
      int16_t* acc = accumulators_[perspectiveIndex].data();
      for (size_t i = 0; i < Shape::kBlockSize; i += 4) {
        const size_t offset0 = i * internal::kInt16PerBlock;
        const size_t offset1 = (i + 1) * internal::kInt16PerBlock;
        const size_t offset2 = (i + 2) * internal::kInt16PerBlock;
        const size_t offset3 = (i + 3) * internal::kInt16PerBlock;

        __m256i wSub0_0 = internal::loadVec(&weights_(subIndex0, offset0));
        __m256i wSub0_1 = internal::loadVec(&weights_(subIndex0, offset1));
        __m256i wSub0_2 = internal::loadVec(&weights_(subIndex0, offset2));
        __m256i wSub0_3 = internal::loadVec(&weights_(subIndex0, offset3));

        __m256i wSub1_0 = internal::loadVec(&weights_(subIndex1, offset0));
        __m256i wSub1_1 = internal::loadVec(&weights_(subIndex1, offset1));
        __m256i wSub1_2 = internal::loadVec(&weights_(subIndex1, offset2));
        __m256i wSub1_3 = internal::loadVec(&weights_(subIndex1, offset3));

        __m256i wAdd_0 = internal::loadVec(&weights_(addIndex, offset0));
        __m256i wAdd_1 = internal::loadVec(&weights_(addIndex, offset1));
        __m256i wAdd_2 = internal::loadVec(&weights_(addIndex, offset2));
        __m256i wAdd_3 = internal::loadVec(&weights_(addIndex, offset3));

        __m256i diff0 = _mm256_sub_epi16(wAdd_0, _mm256_add_epi16(wSub0_0, wSub1_0));
        __m256i diff1 = _mm256_sub_epi16(wAdd_1, _mm256_add_epi16(wSub0_1, wSub1_1));
        __m256i diff2 = _mm256_sub_epi16(wAdd_2, _mm256_add_epi16(wSub0_2, wSub1_2));
        __m256i diff3 = _mm256_sub_epi16(wAdd_3, _mm256_add_epi16(wSub0_3, wSub1_3));

        __m256i acc0 = internal::loadVec(&acc[offset0]);
        __m256i acc1 = internal::loadVec(&acc[offset1]);
        __m256i acc2 = internal::loadVec(&acc[offset2]);
        __m256i acc3 = internal::loadVec(&acc[offset3]);

        acc0 = _mm256_add_epi16(acc0, diff0);
        acc1 = _mm256_add_epi16(acc1, diff1);
        acc2 = _mm256_add_epi16(acc2, diff2);
        acc3 = _mm256_add_epi16(acc3, diff3);

        internal::storeVec(&acc[offset0], acc0);
        internal::storeVec(&acc[offset1], acc1);
        internal::storeVec(&acc[offset2], acc2);
        internal::storeVec(&acc[offset3], acc3);
      }
    }

    std::array<std::span<const uint8_t, Shape::kSize>, kPerspectives> output() noexcept {
      for (size_t p = 0; p < accumulators_.size(); ++p) {
        if (!isComputed[p]) {
          computeOutput(static_cast<uint32_t>(p));
        }
      }
      return { output_[0], output_[1] };
    }
  };

  // Dense Layer using int8_t weights, int32_t biases, and uint8_t activations via VPMADDUBSW.
  template <size_t neuronNum, size_t previousNeuronNum>
  class DenseLayer {
    using Shape = internal::Shape<neuronNum, previousNeuronNum>;
    static_assert(neuronNum % 8 == 0, "DenseLayer requires neuronNum to be a multiple of 8");

    alignas(32) std::array<int32_t, neuronNum> biases_{};
    alignas(32) Matrix<int8_t, neuronNum, Shape::kInputSize> weights_{};
    alignas(32) std::array<uint8_t, neuronNum> output_{};

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
      const __m256i activated = internal::activateSCReLU(sumWithBiases);

      const __m128i activatedLow = _mm256_castsi256_si128(activated);
      const __m128i activatedHigh = _mm256_extracti128_si256(activated, 1);
      const __m128i pack16 = _mm_packus_epi32(activatedLow, activatedHigh);
      const __m128i pack8 = _mm_packus_epi16(pack16, pack16);
      const uint64_t val8 = static_cast<uint64_t>(_mm_cvtsi128_si64(pack8));
      *reinterpret_cast<uint64_t*>(&output_[row]) = val8;
    }

  public:
    DenseLayer() = default;

    std::span<int32_t, neuronNum> biases() noexcept {
      return std::span<int32_t, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<int8_t, neuronNum * previousNeuronNum> weights() noexcept {
      static_assert(Shape::kInputSize == previousNeuronNum);
      return std::span<int8_t, neuronNum * previousNeuronNum>(
        weights_.data(), neuronNum * previousNeuronNum);
    }

    void feedFull(std::span<const uint8_t, Shape::kInputSize> inputs) noexcept {
      const __m256i ones = _mm256_set1_epi16(1);

      for (size_t row = 0; row < neuronNum; row += 8) {
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
    }

    // Zero-copy dual-span feedFull for two contiguous halves (e.g. White + Black accumulators).
    void feedFull(std::span<const uint8_t, Shape::kInputSize / 2> in0,
                  std::span<const uint8_t, Shape::kInputSize / 2> in1) noexcept {
      static_assert(Shape::kInputSize % 2 == 0, "Input size must be even for dual-span feedFull");
      constexpr size_t HalfSize = Shape::kInputSize / 2;
      const __m256i ones = _mm256_set1_epi16(1);

      for (size_t row = 0; row < neuronNum; row += 8) {
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

    std::span<const uint8_t, neuronNum> output() const noexcept {
      return output_;
    }
  };

  // Bucketed output layer computing final centipawn score.
  template <size_t previousNeuronNum, size_t bucketNum>
  class OutputLayer {
    using Shape = internal::Shape<1, previousNeuronNum>;
    static constexpr size_t kWeightsPerVector = sizeof(__m256i) / sizeof(int16_t); // 16
    static constexpr size_t kInputsPerStep = 2 * kWeightsPerVector; // 32
    static constexpr int32_t kOutputShift = internal::kOutputShift;
    static constexpr int32_t kOutputRounding = internal::kOutputRounding;

    alignas(32) std::array<std::array<int16_t, Shape::kInputSize>, bucketNum> weights_{};
    std::array<int32_t, bucketNum> biases_{};

  public:
    OutputLayer() = default;

    std::span<int32_t, bucketNum> biases() noexcept {
      return std::span<int32_t, bucketNum>(biases_.data(), bucketNum);
    }

    std::span<int16_t, bucketNum * previousNeuronNum> weights() noexcept {
      return std::span<int16_t, bucketNum * previousNeuronNum>(
        reinterpret_cast<int16_t*>(weights_.data()), bucketNum * previousNeuronNum);
    }

    int32_t feedFull(std::span<const uint8_t, Shape::kInputSize> inputs, uint32_t bucket) const noexcept {
      static_assert(Shape::kInputSize % kInputsPerStep == 0, "OutputLayer requires input size to be a multiple of 32");

      bucket = std::min<uint32_t>(bucket, static_cast<uint32_t>(bucketNum - 1));
      const auto& bWeights = weights_[bucket];

      __m256i sumVec = _mm256_setzero_si256();
      for (size_t col = 0; col < Shape::kInputSize; col += kInputsPerStep) {
        const __m256i input256 = internal::loadVec(inputs.data() + col);
        const __m128i inputLow128 = _mm256_castsi256_si128(input256);
        const __m128i inputHigh128 = _mm256_extracti128_si256(input256, 1);

        const __m256i inputLow16 = _mm256_cvtepu8_epi16(inputLow128);
        const __m256i inputHigh16 = _mm256_cvtepu8_epi16(inputHigh128);

        const __m256i weightLow = internal::loadVec(bWeights.data() + col);
        const __m256i weightHigh = internal::loadVec(bWeights.data() + col + kWeightsPerVector);

        const __m256i productLow = _mm256_madd_epi16(inputLow16, weightLow);
        const __m256i productHigh = _mm256_madd_epi16(inputHigh16, weightHigh);

        sumVec = _mm256_add_epi32(sumVec, _mm256_add_epi32(productLow, productHigh));
      }
      const int32_t total = horizontalAdd32(sumVec) + biases_[bucket];

      // Descale by 2048 (2^11) directly to centipawns with nearest-integer rounding (+1024).
      return (total + kOutputRounding) >> kOutputShift;
    }
  };
}
