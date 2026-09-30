#pragma once
#include <functional>
#include <span>
#include <string_view>

namespace bb::uci {
  class UciProtocol {
  public:
    using SimpleCallback = std::function<void()>;
    using PositionCallback = std::function<void(std::string_view, std::span<const std::string_view>)>;
    using GoCallback = std::function<void(std::span<const std::string_view>)>;

    explicit UciProtocol(
      SimpleCallback onUci, 
      SimpleCallback onIsReady,
      SimpleCallback onNewGame, 
      PositionCallback onPosition,
      GoCallback onGo, 
      SimpleCallback onQuit);

    void send(std::string_view line);

  private:
    SimpleCallback onUci_;
    SimpleCallback onIsReady_;
    SimpleCallback onNewGame_;
    PositionCallback onPosition_;
    GoCallback onGo_;
    SimpleCallback onQuit_;
  };
}
