#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace nnue {
  inline constexpr float kLeakyReluSlope = 0.04f;

  inline float leakyRelu(float value) noexcept {
    return value >= 0.0f ? value : kLeakyReluSlope * value;
  }

  inline __m256 leakyRelu(__m256 value) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    const __m256 slope = _mm256_set1_ps(kLeakyReluSlope);
    return _mm256_add_ps(_mm256_max_ps(zero, value),
                         _mm256_mul_ps(slope, _mm256_min_ps(zero, value)));
  }

  inline float linear(float value) noexcept {
    return value;
  }

  inline float scaledSigmoid(float x) {
    const float scaled = x / 400.0f;
    if (scaled >= 0.0f) {
      const float exponential = std::exp(-scaled);
      return 1.0f / (1.0f + exponential);
    }
    const float exponential = std::exp(scaled);
    return exponential / (1.0f + exponential);
  }

  inline int32_t scaledSigmoidToCentipawns(float probability) {
    probability = std::clamp(probability, 1e-6f, 1.0f - 1e-6f);
    return static_cast<int32_t>(std::lround(400.0f * std::log(probability / (1.0f - probability))));
  }
}
