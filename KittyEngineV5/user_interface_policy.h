#pragma once
#include <concepts>

namespace bb::user_interface {
    template <typename Policy, typename BoardType>
    concept UserInterfacePolicy = requires(Policy& p, BoardType& board) {
      { p.run(board) } -> std::same_as<void>;
      { p.render(board) } -> std::same_as<void>;
    };
}
