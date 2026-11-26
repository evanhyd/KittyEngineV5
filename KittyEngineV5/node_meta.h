#pragma once
#include "bitboard.h"

namespace bb::searching {
  struct NodeMeta {
    enum class NodeType {
      Internal,
      Quiescence,
    };

    const Side side;
    const NodeType type;
    const bool isPV;
    const bool canNullMove;
    const bool isSingularSearch;

    consteval NodeMeta(Side side, NodeType type, bool isPV, bool canNullMove, bool isSingularSearch) noexcept
      : side(side), type(type), isPV(isPV), canNullMove(canNullMove), isSingularSearch(isSingularSearch) {}
    consteval NodeMeta flip() const noexcept { return NodeMeta{ getOtherSide(side), type, isPV, canNullMove, isSingularSearch }; }
    consteval NodeMeta toQuiescence() const noexcept { return NodeMeta{ side, NodeType::Quiescence, false, false, false }; }
    consteval NodeMeta setPV(bool pv) const noexcept { return NodeMeta{ side, type, pv, canNullMove, isSingularSearch }; }
    consteval NodeMeta setCanNullMove(bool enable) const noexcept { return NodeMeta{ side, type, isPV, enable, isSingularSearch }; }
    consteval NodeMeta setSingularSearch(bool enable) const noexcept { return NodeMeta{ side, type, isPV, canNullMove, enable }; }
    consteval bool isInternal() const noexcept { return type == NodeType::Internal; }
    consteval bool isQuiescence() const noexcept { return type == NodeType::Quiescence; }
  };
}
