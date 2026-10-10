#pragma once
#include "build_config.h"
#if KITTY_ENABLE_SYZYGY
#include "boardstate.h"
#include <algorithm>
#include <optional>
#include <string>
#include <string_view>

namespace bb::tablebase {
  inline constexpr int kWinScore = 30'000;
  enum class Wdl { Loss = -2, BlessedLoss = -1, Draw = 0, CursedWin = 1, Win = 2 };
  constexpr int outcome(Wdl wdl) noexcept {
    return wdl == Wdl::Win ? 1 : wdl == Wdl::Loss ? -1 : 0;
  }

  struct RootResult {
    MoveList bestMoves;
    // The DTZ rounding boundary cannot certify an exact outcome.
    std::optional<int> outcome;
  };

  // Own exactly one Fathom session per process. Search borrows this service.
  // Configure/free only between searches; no probe uses a virtual dispatch.
  class Service {
    std::string path_;
    unsigned largest_ = 0;
    unsigned limit_ = 5;
    int probeDepth_ = 1;
    bool initialized_ = false;
  public:
    Service();
    ~Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;
    Service(Service&&) = delete;
    Service& operator=(Service&&) = delete;

    std::string setOption(std::string_view name, std::string_view value);
    bool enabled() const noexcept { return largest_ != 0 && limit_ != 0; }
    unsigned largest() const noexcept { return largest_; }
    unsigned limit() const noexcept { return std::min(limit_, largest_); }
    int probeDepth() const noexcept { return probeDepth_; }
    const std::string& path() const noexcept { return path_; }

    bool covers(const BoardState& state) const noexcept {
      return state.getCastlingRights() == 0 &&
        countPiece(state.getOccupancy(White) | state.getOccupancy(Black)) <= limit();
    }
    bool canProbeWdl(const BoardState& state, int remainingDepth) const noexcept {
      return state.getHalfmoveClock() == 0 && remainingDepth >= probeDepth_ && covers(state);
    }
    std::optional<Wdl> probeWdl(const BoardState& state) const;
    std::optional<RootResult> rankRoot(const BoardState& state, const MoveList& legalMoves, bool hasRepeated) const;
  };
}
#endif
