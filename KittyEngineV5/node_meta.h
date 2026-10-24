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

    consteval NodeMeta(Side side, NodeType type, bool isPV, bool canNullMove) noexcept
      : side(side), type(type), isPV(isPV), canNullMove(canNullMove) {}
    consteval NodeMeta flip() const noexcept { return NodeMeta{ getOtherSide(side), type, isPV, canNullMove }; }
    consteval NodeMeta enableNullMove() const noexcept { return NodeMeta{ side, type, isPV, true }; }
    consteval NodeMeta disableNullMove() const noexcept { return NodeMeta{ side, type, isPV, false }; }
    consteval NodeMeta toQuiescence() const noexcept { return NodeMeta{ side, NodeType::Quiescence, false, false }; }
    consteval NodeMeta setPV(bool pv) const noexcept { return NodeMeta{ side, type, pv, canNullMove }; }
    consteval bool isInternal() const noexcept { return type == NodeType::Internal; }
    consteval bool isQuiescence() const noexcept { return type == NodeType::Quiescence; }
  };
}
