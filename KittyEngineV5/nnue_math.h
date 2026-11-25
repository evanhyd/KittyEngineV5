#pragma once
#include <cstdint>
#include <immintrin.h>

namespace nnue {
  [[nodiscard]] inline int32_t horizontalAdd32(__m256i v) noexcept {
    const __m128i low = _mm256_castsi256_si128(v);
    const __m128i high = _mm256_extracti128_si256(v, 1);
    const __m128i sum128 = _mm_add_epi32(low, high);
    const __m128i s64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    const __m128i s32 = _mm_add_epi32(s64, _mm_srli_si128(s64, 4));
    return _mm_cvtsi128_si32(s32);
  }

  // Widening horizontal sum of 8 signed int32_t lanes into int64_t (immune to 32-bit sum overflow).
  [[nodiscard]] inline int64_t horizontalAdd64(__m256i v) noexcept {
    const __m256i lo64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    const __m256i hi64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    const __m256i sum4 = _mm256_add_epi64(lo64, hi64);
    const __m128i sum2 = _mm_add_epi64(_mm256_castsi256_si128(sum4), _mm256_extracti128_si256(sum4, 1));
    const __m128i sum1 = _mm_add_epi64(sum2, _mm_srli_si128(sum2, 8));
    return _mm_cvtsi128_si64(sum1);
  }
}
