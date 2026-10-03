#pragma once
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>

namespace bb::uci {
  class UciProtocol {
  public:
    using SimpleCallback = std::function<void()>;
    using PositionCallback = std::function<void(std::string_view, std::span<const std::string_view>)>;
    using GoCallback = std::function<void(std::span<const std::string_view>)>;
    using PlayCallback = std::function<void(std::string_view)>;
    using PerftCallback = std::function<void(uint32_t, bool)>;

    explicit UciProtocol(
      SimpleCallback onUci, 
      SimpleCallback onIsReady,
      SimpleCallback onNewGame, 
      PositionCallback onPosition,
      GoCallback onGo, 
      SimpleCallback onQuit,
      PlayCallback onPlay = {},
      PerftCallback onPerft = {});

    void send(std::string_view line);

  private:
    SimpleCallback onUci_;
    SimpleCallback onIsReady_;
    SimpleCallback onNewGame_;
    PositionCallback onPosition_;
    GoCallback onGo_;
    SimpleCallback onQuit_;
    PlayCallback onPlay_;
    PerftCallback onPerft_;
  };
}
