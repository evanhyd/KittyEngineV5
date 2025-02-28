#pragma once
#include "board.h"
#include <chrono>
#include <iostream>

class PerftDriver {
public:
  struct Result {
    uint64_t nodes;
    uint64_t captures;
    uint64_t enpassants;
    uint64_t castles;
    uint64_t promotions;
  };

  template <bool verbose>
  Result perft(BoardState state, uint32_t depth) {
    using namespace std::chrono;

    // Measure
    Result result{};
    auto start = high_resolution_clock::now();
    if (state.getColor() == kWhite) {
      perftImpl<kWhite, verbose>(state, depth, result);
    } else {
      perftImpl<kBlack, verbose>(state, depth, result);
    }
    auto end = high_resolution_clock::now();


    // Report
    auto ms = duration_cast<milliseconds>(end - start);
    uint64_t knps = (ms.count() > 0 ? static_cast<uint64_t>(static_cast<double>(result.nodes) / ms.count()) : result.nodes);
    std::cout << std::format("depth {}, nodes {}, time {}, speed {} knps\n", depth, result.nodes, ms, knps);
    if constexpr (verbose) {
      std::cout << std::format("    captures {} enpassants {} castles {} promotions {}\n",
                               result.captures, result.enpassants, result.castles, result.promotions);
    }

    return result;
  }

private:
  template <Color our, bool verbose>
  void perftImpl(BoardState state, uint32_t depth, Result& result) {
    constexpr Color their = (our == kWhite ? kBlack : kWhite);

    MoveList moveList{};
    state.getLegalMove<our>(moveList);
    for (Move move : moveList) {
      if (depth > 1) {
        perftImpl<their, verbose>(Board::makeMove<our>(state, move), depth - 1, result);
      } else {
        ++result.nodes;
      }
    }
  }
};
