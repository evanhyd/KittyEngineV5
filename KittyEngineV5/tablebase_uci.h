#pragma once
#include "uci_protocol.h"
#include <utility>

namespace bb::uci {
  // Extend option handling without changing the original protocol's layout.
  class TablebaseProtocol : public UciProtocol {
  public:
    using OptionCallback = std::function<void(std::string_view, std::string_view)>;

    template <typename... Callbacks>
    TablebaseProtocol(OptionCallback onOption, Callbacks&&... callbacks)
      : UciProtocol(std::forward<Callbacks>(callbacks)...), onOption_(std::move(onOption)) {}

    void send(std::string_view line);

  private:
    OptionCallback onOption_;
  };

  template <bool Enabled, typename OptionCallback, typename... Callbacks>
  auto makeProtocol(OptionCallback&& onOption, Callbacks&&... callbacks) {
    if constexpr (Enabled) {
      return TablebaseProtocol(std::forward<OptionCallback>(onOption), std::forward<Callbacks>(callbacks)...);
    } else {
      return UciProtocol(std::forward<Callbacks>(callbacks)...);
    }
  }
}
