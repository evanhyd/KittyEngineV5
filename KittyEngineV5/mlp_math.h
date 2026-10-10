#pragma once
#include <algorithm>
#include <cmath>

namespace mlp {
  inline float relu(float x) {
    return std::max(0.0f, x);
  }

  inline float crelu(float x) {
    return std::clamp(x, 0.0f, 1.0f);
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

}
