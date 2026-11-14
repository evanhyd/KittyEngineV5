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
}
