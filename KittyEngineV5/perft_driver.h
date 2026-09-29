#pragma once
#include "boardstate.h"
#include <chrono>
#include <iostream>

namespace perft {
  struct Config {
    bool isParallel;
    bool isBulkCount;
    bool isDetailed;
  };

  struct Result {
    uint64_t nodes;
    uint64_t captures;
    uint64_t enpassants;
    uint64_t castles;
    uint64_t promotions;
  };

  template <Config config, bb::Color ally>
  inline void countChildren(bb::BoardState& state, uint32_t depth, Result& result) {
    bb::MoveList moves;
    state.generateMoves<ally>(moves);
    if (depth == 1) {
      result.nodes += moves.size();
      if constexpr (config.isDetailed) {
        for (const bb::Move& move : moves) {
          result.captures += move.isCapture() || move.isEnpassant();
          result.enpassants += move.isEnpassant();
          result.castles += move.isCastling();
          result.promotions += move.getPromotedPieceType() != bb::kNoPiece;
        }
      }
      return;
    }
    for (const bb::Move& move : moves) {
      const bb::MoveUndo undo = state.makeMove<ally>(move);
      countChildren<config, bb::getOtherColor(ally)>(state, depth - 1, result);
      state.unmakeMove<ally>(move, undo);
    }
  }

  template <Config config>
  inline Result countPerft(const bb::BoardState& position, bb::Color sideToMove, uint32_t depth) {
    static_assert(!(config.isBulkCount && config.isDetailed), "bulk counting is incompatible with detailed perft");
    Result result{};
    if (depth == 0) {
      result.nodes = 1;
      return result;
    }
    bb::BoardState state = position;
    if (sideToMove == bb::kWhite) {
      countChildren<config, bb::kWhite>(state, depth, result);
    } else {
      countChildren<config, bb::kBlack>(state, depth, result);
    }
    return result;
  }

  template <Config config, bool canPrint = true>
  inline Result runPerft(const bb::BoardState& state, bb::Color sideToMove, uint32_t depth) {
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
    const Result result = countPerft<config>(state, sideToMove, depth);
    const auto ms = duration_cast<milliseconds>(high_resolution_clock::now() - start);
    if constexpr (canPrint) {
      const uint64_t knps = ms.count() > 0 ? result.nodes / ms.count() : result.nodes;
      std::cout << "depth " << depth << ", nodes " << result.nodes << ", time " << ms.count()
                << " ms, speed " << knps << " knps\n";
      if constexpr (config.isDetailed) {
        std::cout << "    captures " << result.captures << " enpassants " << result.enpassants
                  << " castles " << result.castles << " promotions " << result.promotions << '\n';
      }
    }
    return result;
  }
}
