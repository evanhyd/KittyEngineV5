#pragma once
#include "tablebase.h"

namespace bb::tablebase {
  // Empty bases keep disabled search objects and contexts free of probe state.
  template <bool Enabled>
  struct SearchState {};

  template <>
  struct SearchState<true> {
    Service* tablebases_ = nullptr;
    std::optional<RootResult> rootTablebase_;
    bool rootProbed_ = false;
    uint64_t lastTablebaseHits_ = 0;
  };

  template <bool Enabled>
  struct ProbeCounter {};

  template <>
  struct ProbeCounter<true> {
    uint64_t tablebaseHits = 0;
  };
}
