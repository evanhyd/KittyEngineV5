#pragma once
#include <cstdint>
#include <immintrin.h>

namespace nnue {
  [[nodiscard]] inline int32_t horizontalAdd32(__m256i v) noexcept {
    const __m128i low = _mm256_castsi256_si128(v);
    const __m128i high = _mm256_extracti128_si256(v, 1);
    const __m128i sum128 = _mm_add_epi32(low, high);
    const __m128i hadd1 = _mm_hadd_epi32(sum128, sum128);
    const __m128i hadd2 = _mm_hadd_epi32(hadd1, hadd1);
    return _mm_cvtsi128_si32(hadd2);
  }
}
