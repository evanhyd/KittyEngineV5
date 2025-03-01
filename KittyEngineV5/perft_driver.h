#pragma once
#include "board.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <execution>
#include <iostream>

class PerftDriver {
public:
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

  template <Config config, bool canPrint = true>
  Result perft(BoardState state, uint32_t depth) {
    static_assert(!(config.isBulkCount && config.isDetailed), "bulk counting is incompatiable with detailed perft");
    using namespace std::chrono;

    Result result{};
    auto start = high_resolution_clock::now();
    switch (depth) {
    case 1: state.getColor() == kWhite ? perftImpl<config, kWhite, 1>(state, result) : perftImpl<config, kBlack, 1>(state, result); break;
    case 2: state.getColor() == kWhite ? perftImpl<config, kWhite, 2>(state, result) : perftImpl<config, kBlack, 2>(state, result); break;
    case 3: state.getColor() == kWhite ? perftImpl<config, kWhite, 3>(state, result) : perftImpl<config, kBlack, 3>(state, result); break;
    case 4: state.getColor() == kWhite ? perftImpl<config, kWhite, 4>(state, result) : perftImpl<config, kBlack, 4>(state, result); break;
    case 5: state.getColor() == kWhite ? perftImpl<config, kWhite, 5>(state, result) : perftImpl<config, kBlack, 5>(state, result); break;
    case 6: state.getColor() == kWhite ? perftImpl<config, kWhite, 6>(state, result) : perftImpl<config, kBlack, 6>(state, result); break;
    case 7: state.getColor() == kWhite ? perftImpl<config, kWhite, 7>(state, result) : perftImpl<config, kBlack, 7>(state, result); break;
    case 8: state.getColor() == kWhite ? perftImpl<config, kWhite, 8>(state, result) : perftImpl<config, kBlack, 8>(state, result); break;
    case 9: state.getColor() == kWhite ? perftImpl<config, kWhite, 9>(state, result) : perftImpl<config, kBlack, 9>(state, result); break;
    case 10: state.getColor() == kWhite ? perftImpl<config, kWhite, 10>(state, result) : perftImpl<config, kBlack, 10>(state, result); break;
    case 11: state.getColor() == kWhite ? perftImpl<config, kWhite, 11>(state, result) : perftImpl<config, kBlack, 11>(state, result); break;
    case 12: state.getColor() == kWhite ? perftImpl<config, kWhite, 12>(state, result) : perftImpl<config, kBlack, 12>(state, result); break;
    case 13: state.getColor() == kWhite ? perftImpl<config, kWhite, 13>(state, result) : perftImpl<config, kBlack, 13>(state, result); break;
    default: break;
    }
    auto end = high_resolution_clock::now();

    if constexpr (canPrint) {
      auto ms = duration_cast<milliseconds>(end - start);
      uint64_t knps = (ms.count() > 0 ? static_cast<uint64_t>(static_cast<double>(result.nodes) / ms.count()) : result.nodes);
      std::cout << std::format("depth {}, nodes {}, time {}, speed {} knps\n", depth, result.nodes, ms, knps);
      if constexpr (config.isDetailed) {
        std::cout << std::format("    captures {} enpassants {} castles {} promotions {}\n",
                                 result.captures, result.enpassants, result.castles, result.promotions);
      }
    }

    return result;
  }

private:
  template <Config config, Color our, size_t depth>
  void perftImpl(BoardState state, Result& result) {
    constexpr Color their = (our == kWhite ? kBlack : kWhite);

    // Generate legal moves.
    MoveList moveList{};
    state.getLegalMove<our>(moveList);

    // Leaf node.
    if constexpr (depth <= 1) {
      if constexpr (config.isBulkCount) {
        result.nodes += moveList.size();
      } else {
        for (Move move : moveList) {
          // May be pick up other details.
          ++result.nodes;
        }
      }
    } else if constexpr (config.isParallel) {

      // Multi threads searching.
      std::atomic<uint64_t> totalNodes = 0;
      std::for_each(std::execution::par_unseq, moveList.begin(), moveList.end(), [&](Move move) {
        Result subResult{};
        perftImpl<Config{false, config.isBulkCount, config.isDetailed}, their, depth - 1>(
          Board::makeMove<our>(state, move), 
          subResult
        );
        totalNodes += subResult.nodes;
      });
      result.nodes = totalNodes;
    } else {

      // Single thread searching.
      for (Move move : moveList) {
        perftImpl<Config{false, config.isBulkCount, config.isDetailed}, their, depth - 1>(
          Board::makeMove<our>(state, move), 
          result
        );
      }
    }
  }
};
