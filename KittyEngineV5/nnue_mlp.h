#pragma once
#include "nnue_layer.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace nnue {
  class NnueModel {
  public:
    static constexpr size_t kInputSize = 12'288;      // 16 king buckets * 768 piece planes
    static constexpr size_t kAccumulatorSize = 768;   // 768 neurons per perspective (384 pairwise pairs)
    static constexpr size_t kPairwiseSize = 768;      // 384 STM + 384 NTM
    static constexpr size_t kHidden1Size = 16;
    static constexpr size_t kHidden2Size = 32;
    static constexpr size_t kOutputBuckets = 8;

  private:
    FeatureTransformer<kAccumulatorSize, kInputSize> featureTransformer_;
    BucketedL1Layer<kHidden1Size, kPairwiseSize, kOutputBuckets> l1_;
    BucketedL2Layer<kHidden2Size, kHidden1Size, kOutputBuckets> l2_;
    BucketedOutputLayer<kHidden2Size, kOutputBuckets> output_;

  public:
    FeatureTransformer<kAccumulatorSize, kInputSize>& featureTransformer() noexcept {
      return featureTransformer_;
    }

    const FeatureTransformer<kAccumulatorSize, kInputSize>& featureTransformer() const noexcept {
      return featureTransformer_;
    }

    BucketedL1Layer<kHidden1Size, kPairwiseSize, kOutputBuckets>& firstHidden() noexcept {
      return l1_;
    }

    BucketedL2Layer<kHidden2Size, kHidden1Size, kOutputBuckets>& secondHidden() noexcept {
      return l2_;
    }

    BucketedOutputLayer<kHidden2Size, kOutputBuckets>& outputLayer() noexcept {
      return output_;
    }

    [[nodiscard]] int32_t evaluate(const int16_t* stmAcc, const int16_t* ntmAcc, uint32_t bucket) const noexcept {
      bucket = std::min<uint32_t>(bucket, static_cast<uint32_t>(kOutputBuckets - 1));
      alignas(32) std::array<uint8_t, kPairwiseSize> pairwise;
      alignas(16) std::array<uint8_t, kHidden1Size> h1;
      alignas(32) std::array<uint8_t, kHidden2Size> h2;

      pairwiseCReLU<kAccumulatorSize>(stmAcc, ntmAcc, pairwise.data());
      l1_.evaluate(pairwise.data(), bucket, h1.data());
      l2_.evaluate(h1.data(), bucket, h2.data());
      return output_.evaluate(h2.data(), bucket);
    }
  };
}
