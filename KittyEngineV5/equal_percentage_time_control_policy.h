#pragma once
#include "time_control_policy.h"
#include <cstdint>
#include <chrono>

namespace bb::time_control {
  class EqualPercentageTimeControlPolicy {
    float percentage_;
    bool isTimeControlSet_ = false;
    std::array<std::chrono::milliseconds, kSideSize> searchingTime{};
    std::chrono::high_resolution_clock::time_point start_{};
    std::chrono::high_resolution_clock::time_point lastSearchStart_{};

  public:
    explicit EqualPercentageTimeControlPolicy(float percentage) noexcept 
    : percentage_(percentage) {
    }

    void set(const std::optional<TimeControl>& timeControl) {
      using namespace std::chrono;

      isTimeControlSet_ = timeControl.has_value();
      if (isTimeControlSet_) {
        start_ = high_resolution_clock::now();
        lastSearchStart_ = start_;

        // Calculate the allowed thinking time.
        searchingTime[White] = std::max(duration_cast<milliseconds>(timeControl->wtime.value_or(0ms) * percentage_), timeControl->winc.value_or(0ms));
        searchingTime[Black] = std::max(duration_cast<milliseconds>(timeControl->btime.value_or(0ms) * percentage_), timeControl->binc.value_or(0ms));
      }
    }

    bool shouldContinue(Side side) {
      if (!isTimeControlSet_) {
        return true;
      }

      const auto now = std::chrono::high_resolution_clock::now();
      const auto usedTime = now - start_;
      const auto remainingTime = searchingTime[side] - usedTime;
      const auto lastSearchTime = now - lastSearchStart_;

      // Not enough time for higher depth, stop searching.
      if (lastSearchTime * 3 >= remainingTime) {
        isTimeControlSet_ = false;
        return false;
      }

      // Update the clock.
      lastSearchStart_ = now;
      return true;
    }
  };

  static_assert(TimeControlPolicy<EqualPercentageTimeControlPolicy>, "Must satisfy TimeControlPolicy");
}
