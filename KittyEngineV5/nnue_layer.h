#pragma once
#include "nnue_matrix.h"
#include "nnue_math.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <span>

namespace nnue {
  namespace internal {
    inline constexpr size_t kInt16PerBlock = sizeof(__m256i) / sizeof(int16_t); // 16
    inline constexpr size_t kInt8PerBlock  = sizeof(__m256i) / sizeof(int8_t);  // 32
    inline constexpr int16_t kQA = 255;
    inline constexpr int16_t kQB = 64;
    inline constexpr int32_t kActMax = 127;
    inline constexpr int32_t kDenseClamp = kActMax * kQB; // 8128
    inline constexpr int32_t kDenseScaleShift = 6;        // / 64 -> [0, 127]
    inline constexpr int32_t kDenseSquareShift = 7;       // / 128 -> [0, 126]
    inline constexpr int32_t kEvalScale = 400;
    inline constexpr int32_t kPackPermute = _MM_SHUFFLE(3, 1, 2, 0);

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
      const __m256i clampMax = _mm256_set1_epi32(kDenseClamp);
      const __m256i maxAct = _mm256_set1_epi32(kActMax);

      const __m256i clamped = _mm256_min_epi32(_mm256_max_epi32(sumWithBiases, zero), clampMax);
      const __m256i scaled = _mm256_srli_epi32(clamped, kDenseScaleShift);
      const __m256i squared = _mm256_mullo_epi32(scaled, scaled);
      return _mm256_min_epi32(_mm256_srli_epi32(squared, kDenseSquareShift), maxAct);
    }

    // Packs two 8-lane int32_t activated vectors ([0..7], [8..15]) into 16 contiguous uint8_t lanes.
    [[nodiscard]] inline __m128i pack16U8(__m256i act0, __m256i act1) noexcept {
      const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(act0, act1), kPackPermute);
      return _mm_packus_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
    }
  }

  // Fused Pairwise CReLU: clamps each int16_t accumulator half to [0, 255], multiplies pairs,
  // right-shifts by 9 (yielding [0, 127]), and packs STM || NTM into uint8_t[accumSize].
  template <size_t accumSize>
  inline void pairwiseCReLU(const int16_t* stmAcc, const int16_t* ntmAcc, uint8_t* out) noexcept {
    static_assert(accumSize % 64 == 0, "pairwiseCReLU requires accumSize to be a multiple of 64");
    constexpr size_t kHalf = accumSize / 2;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qaVec = _mm256_set1_epi16(internal::kQA);

    auto runHalf = [&](const int16_t* acc, uint8_t* dst) noexcept {
      for (size_t i = 0; i < kHalf; i += 2 * internal::kInt16PerBlock) {
        const size_t i1 = i + internal::kInt16PerBlock;

        const __m256i lo0 = _mm256_min_epi16(_mm256_max_epi16(internal::loadVec(&acc[i]), zero), qaVec);
        const __m256i hi0 = _mm256_min_epi16(_mm256_max_epi16(internal::loadVec(&acc[kHalf + i]), zero), qaVec);
        const __m256i p0 = _mm256_mulhi_epu16(_mm256_slli_epi16(lo0, 7), hi0);

        const __m256i lo1 = _mm256_min_epi16(_mm256_max_epi16(internal::loadVec(&acc[i1]), zero), qaVec);
        const __m256i hi1 = _mm256_min_epi16(_mm256_max_epi16(internal::loadVec(&acc[kHalf + i1]), zero), qaVec);
        const __m256i p1 = _mm256_mulhi_epu16(_mm256_slli_epi16(lo1, 7), hi1);

        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(p0, p1), internal::kPackPermute);
        internal::storeVec(&dst[i], packed);
      }
    };

    runHalf(stmAcc, out);
    runHalf(ntmAcc, out + kHalf);
  }

  // Stateless AVX2 feature transformer (QA = 255) for ply-indexed accumulator stacks and Finny tables.
  template <size_t neuronNum, size_t previousNeuronNum>
  class FeatureTransformer {
    static constexpr size_t kBlockSize = neuronNum / internal::kInt16PerBlock;
    static_assert(neuronNum % internal::kInt16PerBlock == 0, "FeatureTransformer requires neuronNum to be a multiple of 16");
    static_assert(kBlockSize % 8 == 0, "FeatureTransformer requires block size to be a multiple of 8 for register tiling");

    alignas(32) std::array<int16_t, neuronNum> biases_{};
    alignas(32) Matrix<int16_t, previousNeuronNum, neuronNum> weights_{};

  public:
    FeatureTransformer() = default;

    std::span<int16_t, neuronNum> biases() noexcept {
      return std::span<int16_t, neuronNum>(biases_.data(), neuronNum);
    }

    std::span<int16_t, previousNeuronNum * neuronNum> weights() noexcept {
      return std::span<int16_t, previousNeuronNum * neuronNum>(
        weights_.data(), previousNeuronNum * neuronNum);
    }

    void copyAccumulator(const int16_t* src, int16_t* dst) const noexcept {
      for (size_t i = 0; i < kBlockSize; i += 8) {
        const size_t o0 = (i + 0) * internal::kInt16PerBlock;
        const size_t o1 = (i + 1) * internal::kInt16PerBlock;
        const size_t o2 = (i + 2) * internal::kInt16PerBlock;
        const size_t o3 = (i + 3) * internal::kInt16PerBlock;
        const size_t o4 = (i + 4) * internal::kInt16PerBlock;
        const size_t o5 = (i + 5) * internal::kInt16PerBlock;
        const size_t o6 = (i + 6) * internal::kInt16PerBlock;
        const size_t o7 = (i + 7) * internal::kInt16PerBlock;
        internal::storeVec(&dst[o0], internal::loadVec(&src[o0]));
        internal::storeVec(&dst[o1], internal::loadVec(&src[o1]));
        internal::storeVec(&dst[o2], internal::loadVec(&src[o2]));
        internal::storeVec(&dst[o3], internal::loadVec(&src[o3]));
        internal::storeVec(&dst[o4], internal::loadVec(&src[o4]));
        internal::storeVec(&dst[o5], internal::loadVec(&src[o5]));
        internal::storeVec(&dst[o6], internal::loadVec(&src[o6]));
        internal::storeVec(&dst[o7], internal::loadVec(&src[o7]));
      }
    }

    // Register-tiled multi-feature update: loads src once per 128-element tile into 8 YMM registers,
    // applies all subtractions and additions in registers, and stores dst once.
    void applyUpdates(const int16_t* src, int16_t* dst,
                      std::span<const uint32_t> subs, std::span<const uint32_t> adds) const noexcept {
      for (size_t i = 0; i < kBlockSize; i += 8) {
        const size_t o0 = (i + 0) * internal::kInt16PerBlock;
        const size_t o1 = (i + 1) * internal::kInt16PerBlock;
        const size_t o2 = (i + 2) * internal::kInt16PerBlock;
        const size_t o3 = (i + 3) * internal::kInt16PerBlock;
        const size_t o4 = (i + 4) * internal::kInt16PerBlock;
        const size_t o5 = (i + 5) * internal::kInt16PerBlock;
        const size_t o6 = (i + 6) * internal::kInt16PerBlock;
        const size_t o7 = (i + 7) * internal::kInt16PerBlock;

        __m256i acc0 = internal::loadVec(&src[o0]);
        __m256i acc1 = internal::loadVec(&src[o1]);
        __m256i acc2 = internal::loadVec(&src[o2]);
        __m256i acc3 = internal::loadVec(&src[o3]);
        __m256i acc4 = internal::loadVec(&src[o4]);
        __m256i acc5 = internal::loadVec(&src[o5]);
        __m256i acc6 = internal::loadVec(&src[o6]);
        __m256i acc7 = internal::loadVec(&src[o7]);

        for (uint32_t subIdx : subs) {
          const int16_t* row = &weights_(subIdx, o0);
          acc0 = _mm256_sub_epi16(acc0, internal::loadVec(row + 0 * internal::kInt16PerBlock));
          acc1 = _mm256_sub_epi16(acc1, internal::loadVec(row + 1 * internal::kInt16PerBlock));
          acc2 = _mm256_sub_epi16(acc2, internal::loadVec(row + 2 * internal::kInt16PerBlock));
          acc3 = _mm256_sub_epi16(acc3, internal::loadVec(row + 3 * internal::kInt16PerBlock));
          acc4 = _mm256_sub_epi16(acc4, internal::loadVec(row + 4 * internal::kInt16PerBlock));
          acc5 = _mm256_sub_epi16(acc5, internal::loadVec(row + 5 * internal::kInt16PerBlock));
          acc6 = _mm256_sub_epi16(acc6, internal::loadVec(row + 6 * internal::kInt16PerBlock));
          acc7 = _mm256_sub_epi16(acc7, internal::loadVec(row + 7 * internal::kInt16PerBlock));
        }

        for (uint32_t addIdx : adds) {
          const int16_t* row = &weights_(addIdx, o0);
          acc0 = _mm256_add_epi16(acc0, internal::loadVec(row + 0 * internal::kInt16PerBlock));
          acc1 = _mm256_add_epi16(acc1, internal::loadVec(row + 1 * internal::kInt16PerBlock));
          acc2 = _mm256_add_epi16(acc2, internal::loadVec(row + 2 * internal::kInt16PerBlock));
          acc3 = _mm256_add_epi16(acc3, internal::loadVec(row + 3 * internal::kInt16PerBlock));
          acc4 = _mm256_add_epi16(acc4, internal::loadVec(row + 4 * internal::kInt16PerBlock));
          acc5 = _mm256_add_epi16(acc5, internal::loadVec(row + 5 * internal::kInt16PerBlock));
          acc6 = _mm256_add_epi16(acc6, internal::loadVec(row + 6 * internal::kInt16PerBlock));
          acc7 = _mm256_add_epi16(acc7, internal::loadVec(row + 7 * internal::kInt16PerBlock));
        }

        internal::storeVec(&dst[o0], acc0);
        internal::storeVec(&dst[o1], acc1);
        internal::storeVec(&dst[o2], acc2);
        internal::storeVec(&dst[o3], acc3);
        internal::storeVec(&dst[o4], acc4);
        internal::storeVec(&dst[o5], acc5);
        internal::storeVec(&dst[o6], acc6);
        internal::storeVec(&dst[o7], acc7);
      }
    }

    void feedFull(std::span<const uint32_t> activeIndices, int16_t* dst) const noexcept {
      applyUpdates(biases_.data(), dst, {}, activeIndices);
    }

    // Fused copy + quiet move update (1 sub + 1 add) in a single 4x-unrolled memory pass.
    void copyAndUpdatePair(const int16_t* src, int16_t* dst, uint32_t subIndex, uint32_t addIndex) const noexcept {
      const int16_t* wSub = &weights_(subIndex, 0);
      const int16_t* wAdd = &weights_(addIndex, 0);
      for (size_t i = 0; i < kBlockSize; i += 4) {
        const size_t o0 = (i + 0) * internal::kInt16PerBlock;
        const size_t o1 = (i + 1) * internal::kInt16PerBlock;
        const size_t o2 = (i + 2) * internal::kInt16PerBlock;
        const size_t o3 = (i + 3) * internal::kInt16PerBlock;

        const __m256i d0 = _mm256_sub_epi16(internal::loadVec(wAdd + o0), internal::loadVec(wSub + o0));
        const __m256i d1 = _mm256_sub_epi16(internal::loadVec(wAdd + o1), internal::loadVec(wSub + o1));
        const __m256i d2 = _mm256_sub_epi16(internal::loadVec(wAdd + o2), internal::loadVec(wSub + o2));
        const __m256i d3 = _mm256_sub_epi16(internal::loadVec(wAdd + o3), internal::loadVec(wSub + o3));

        internal::storeVec(&dst[o0], _mm256_add_epi16(internal::loadVec(&src[o0]), d0));
        internal::storeVec(&dst[o1], _mm256_add_epi16(internal::loadVec(&src[o1]), d1));
        internal::storeVec(&dst[o2], _mm256_add_epi16(internal::loadVec(&src[o2]), d2));
        internal::storeVec(&dst[o3], _mm256_add_epi16(internal::loadVec(&src[o3]), d3));
      }
    }

    // Fused copy + capture move update (2 subs + 1 add) in a single 4x-unrolled memory pass.
    void copyAndUpdateCapture(const int16_t* src, int16_t* dst, uint32_t subIndex0, uint32_t subIndex1, uint32_t addIndex) const noexcept {
      const int16_t* wSub0 = &weights_(subIndex0, 0);
      const int16_t* wSub1 = &weights_(subIndex1, 0);
      const int16_t* wAdd  = &weights_(addIndex, 0);
      for (size_t i = 0; i < kBlockSize; i += 4) {
        const size_t o0 = (i + 0) * internal::kInt16PerBlock;
        const size_t o1 = (i + 1) * internal::kInt16PerBlock;
        const size_t o2 = (i + 2) * internal::kInt16PerBlock;
        const size_t o3 = (i + 3) * internal::kInt16PerBlock;

        const __m256i s0 = _mm256_add_epi16(internal::loadVec(wSub0 + o0), internal::loadVec(wSub1 + o0));
        const __m256i s1 = _mm256_add_epi16(internal::loadVec(wSub0 + o1), internal::loadVec(wSub1 + o1));
        const __m256i s2 = _mm256_add_epi16(internal::loadVec(wSub0 + o2), internal::loadVec(wSub1 + o2));
        const __m256i s3 = _mm256_add_epi16(internal::loadVec(wSub0 + o3), internal::loadVec(wSub1 + o3));

        const __m256i d0 = _mm256_sub_epi16(internal::loadVec(wAdd + o0), s0);
        const __m256i d1 = _mm256_sub_epi16(internal::loadVec(wAdd + o1), s1);
        const __m256i d2 = _mm256_sub_epi16(internal::loadVec(wAdd + o2), s2);
        const __m256i d3 = _mm256_sub_epi16(internal::loadVec(wAdd + o3), s3);

        internal::storeVec(&dst[o0], _mm256_add_epi16(internal::loadVec(&src[o0]), d0));
        internal::storeVec(&dst[o1], _mm256_add_epi16(internal::loadVec(&src[o1]), d1));
        internal::storeVec(&dst[o2], _mm256_add_epi16(internal::loadVec(&src[o2]), d2));
        internal::storeVec(&dst[o3], _mm256_add_epi16(internal::loadVec(&src[o3]), d3));
      }
    }
  };

  // Bucketed L1 Dense Layer (768 -> 16 x 8 buckets) using Stockfish-style 4-byte NNZ sparse matmul.
  // Weights are stored transposed as [bucket][col_chunk_4][16_neurons][4_bytes] (64 B = 1 cache line/chunk),
  // keeping all 16 output accumulators in 2 YMM registers with zero horizontal transposes.
  template <size_t neuronNum, size_t inputNum, size_t bucketNum>
  class BucketedL1Layer {
    static_assert(neuronNum == 16, "BucketedL1Layer expects neuronNum == 16");
    static_assert(inputNum % internal::kInt8PerBlock == 0, "BucketedL1Layer requires inputNum to be a multiple of 32");
    static constexpr size_t kChunkBytes = 4;
    static constexpr size_t kChunkStride = neuronNum * kChunkBytes; // 64 bytes = 1 cache line

    alignas(32) std::array<std::array<int32_t, neuronNum>, bucketNum> biases_{};
    alignas(64) std::array<std::array<int8_t, neuronNum * inputNum>, bucketNum> weights_{};

  public:
    BucketedL1Layer() = default;

    [[nodiscard]] static constexpr size_t weightIndex(size_t row, size_t col) noexcept {
      return ((col >> 2) * neuronNum + row) * kChunkBytes + (col & 3u);
    }

    std::span<int32_t, neuronNum> biases(size_t bucket) noexcept {
      return std::span<int32_t, neuronNum>(biases_[bucket].data(), neuronNum);
    }

    std::span<int8_t, neuronNum * inputNum> weights(size_t bucket) noexcept {
      return std::span<int8_t, neuronNum * inputNum>(weights_[bucket].data(), neuronNum * inputNum);
    }

    void evaluate(const uint8_t* inputs, uint32_t bucket, uint8_t* out) const noexcept {
      constexpr size_t kNumChunks = inputNum / kChunkBytes; // 192
      const int8_t* bWeights = weights_[bucket].data();
      const int32_t* bBiases = biases_[bucket].data();
      const auto* in32 = reinterpret_cast<const int32_t*>(inputs);
      const __m256i ones = _mm256_set1_epi16(1);

      __m256i acc0 = internal::loadVec(&bBiases[0]);
      __m256i acc1 = internal::loadVec(&bBiases[8]);

      for (size_t c = 0; c < kNumChunks; c += 4) {
        const __m256i in0 = _mm256_set1_epi32(in32[c + 0]);
        const __m256i in1 = _mm256_set1_epi32(in32[c + 1]);
        const __m256i in2 = _mm256_set1_epi32(in32[c + 2]);
        const __m256i in3 = _mm256_set1_epi32(in32[c + 3]);

        const int8_t* w = bWeights + c * kChunkStride;

        const __m256i s0a = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, internal::loadVec(w + 0 * kChunkStride)), ones);
        const __m256i s0b = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, internal::loadVec(w + 0 * kChunkStride + 32)), ones);
        const __m256i s1a = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, internal::loadVec(w + 1 * kChunkStride)), ones);
        const __m256i s1b = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, internal::loadVec(w + 1 * kChunkStride + 32)), ones);
        const __m256i s2a = _mm256_madd_epi16(_mm256_maddubs_epi16(in2, internal::loadVec(w + 2 * kChunkStride)), ones);
        const __m256i s2b = _mm256_madd_epi16(_mm256_maddubs_epi16(in2, internal::loadVec(w + 2 * kChunkStride + 32)), ones);
        const __m256i s3a = _mm256_madd_epi16(_mm256_maddubs_epi16(in3, internal::loadVec(w + 3 * kChunkStride)), ones);
        const __m256i s3b = _mm256_madd_epi16(_mm256_maddubs_epi16(in3, internal::loadVec(w + 3 * kChunkStride + 32)), ones);

        acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(_mm256_add_epi32(s0a, s1a), _mm256_add_epi32(s2a, s3a)));
        acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(_mm256_add_epi32(s0b, s1b), _mm256_add_epi32(s2b, s3b)));
      }

      const __m128i packed = internal::pack16U8(internal::activateSCReLU(acc0), internal::activateSCReLU(acc1));
      _mm_store_si128(reinterpret_cast<__m128i*>(out), packed);
    }
  };

  // Bucketed L2 Dense Layer (16 -> 32 x 8 buckets) with transposed pair-major weights:
  // [bucket][in_pair_2][32_neurons][2_int16s], keeping all 32 outputs in 4 YMM registers
  // with sparse zero-pair skipping and zero horizontal transposes.
  template <size_t neuronNum, size_t inputNum, size_t bucketNum>
  class BucketedL2Layer {
    static_assert(neuronNum == 32, "BucketedL2Layer expects neuronNum == 32");
    static_assert(inputNum == internal::kInt16PerBlock, "BucketedL2Layer expects inputNum == 16");
    static constexpr size_t kPairCount = inputNum / 2;
    static constexpr size_t kPairStride = neuronNum * 2; // 64 int16_t = 128 bytes per input pair

    alignas(32) std::array<std::array<int32_t, neuronNum>, bucketNum> biases_{};
    alignas(32) std::array<std::array<int16_t, neuronNum * inputNum>, bucketNum> weights_{};

  public:
    BucketedL2Layer() = default;

    [[nodiscard]] static constexpr size_t weightIndex(size_t row, size_t col) noexcept {
      return ((col >> 1) * neuronNum + row) * 2 + (col & 1u);
    }

    std::span<int32_t, neuronNum> biases(size_t bucket) noexcept {
      return std::span<int32_t, neuronNum>(biases_[bucket].data(), neuronNum);
    }

    std::span<int16_t, neuronNum * inputNum> weights(size_t bucket) noexcept {
      return std::span<int16_t, neuronNum * inputNum>(weights_[bucket].data(), neuronNum * inputNum);
    }

    void evaluate(const uint8_t* inputs, uint32_t bucket, uint8_t* out) const noexcept {
      const int16_t* bWeights = weights_[bucket].data();
      const int32_t* bBiases = biases_[bucket].data();

      __m256i acc0 = internal::loadVec(&bBiases[0]);
      __m256i acc1 = internal::loadVec(&bBiases[8]);
      __m256i acc2 = internal::loadVec(&bBiases[16]);
      __m256i acc3 = internal::loadVec(&bBiases[24]);

      for (size_t p = 0; p < kPairCount; ++p) {
        const uint32_t pair = static_cast<uint32_t>(inputs[2 * p])
                            | (static_cast<uint32_t>(inputs[2 * p + 1]) << 16);
        if (pair == 0) {
          continue;
        }
        const __m256i inPair = _mm256_set1_epi32(static_cast<int32_t>(pair));
        const int16_t* w = bWeights + p * kPairStride;

        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(inPair, internal::loadVec(w + 0)));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(inPair, internal::loadVec(w + 16)));
        acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(inPair, internal::loadVec(w + 32)));
        acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(inPair, internal::loadVec(w + 48)));
      }

      const __m128i lo16 = internal::pack16U8(internal::activateSCReLU(acc0), internal::activateSCReLU(acc1));
      const __m128i hi16 = internal::pack16U8(internal::activateSCReLU(acc2), internal::activateSCReLU(acc3));
      internal::storeVec(out, _mm256_set_m128i(hi16, lo16));
    }
  };

  // Bucketed L3 Output Layer (32 -> 1 x 8 buckets) computing final centipawns in a single AVX2 pass.
  template <size_t inputNum, size_t bucketNum>
  class BucketedOutputLayer {
    static_assert(inputNum == internal::kInt8PerBlock, "BucketedOutputLayer expects inputNum == 32");

    alignas(32) std::array<std::array<int16_t, inputNum>, bucketNum> weights_{};
    std::array<int32_t, bucketNum> biases_{};

  public:
    BucketedOutputLayer() = default;

    std::span<int32_t, bucketNum> biases() noexcept {
      return std::span<int32_t, bucketNum>(biases_.data(), bucketNum);
    }

    std::span<int16_t, inputNum> weights(size_t bucket) noexcept {
      return std::span<int16_t, inputNum>(weights_[bucket].data(), inputNum);
    }

    [[nodiscard]] int32_t evaluate(const uint8_t* inputs, uint32_t bucket) const noexcept {
      const auto& bWeights = weights_[bucket];
      const __m256i in256 = internal::loadVec(inputs);
      const __m256i inLo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(in256));
      const __m256i inHi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(in256, 1));

      const __m256i prodLo = _mm256_madd_epi16(inLo, internal::loadVec(bWeights.data()));
      const __m256i prodHi = _mm256_madd_epi16(inHi, internal::loadVec(bWeights.data() + internal::kInt16PerBlock));

      const int64_t total = static_cast<int64_t>(horizontalAdd32(_mm256_add_epi32(prodLo, prodHi)))
                          + static_cast<int64_t>(biases_[bucket]);
      constexpr int64_t kDenom = static_cast<int64_t>(internal::kDenseClamp); // 127 * 64 = 8128
      const int64_t scaled = total * internal::kEvalScale;
      return static_cast<int32_t>((scaled >= 0 ? (scaled + kDenom / 2) : (scaled - kDenom / 2)) / kDenom);
    }
  };
}
