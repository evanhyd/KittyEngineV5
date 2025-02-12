#pragma once
#include "board.h"
#include <chrono>

class PerftDriver : public Board {
  struct Result {
    uint64_t nodes;
    uint64_t captures;
    uint64_t enpassants;
    uint64_t castles;
    uint64_t promotions;
  };

  // depth 5, nodes 193690690, time 8649ms, speed 22394 knps
  template <Team attacker, bool verbose>
  void perftImpl(uint32_t depth, Result& result) {
    if (depth == 0) {
      ++result.nodes;
      return;
    }

    constexpr Team defender = (attacker == kWhite ? kBlack : kWhite);
    MoveList moveList{};
    currentState().getPseudoMove<attacker>(moveList);
    for (Move move : moveList) {
      if (tryMakeMove<attacker>(move)) {
        perftImpl<defender, verbose>(depth - 1, result);
        popState();

        if constexpr (verbose) {
          if (depth == 1) {
            if (move.isCaptured()) {
              ++result.captures;
            }
            if (move.isCastling()) {
              ++result.castles;
            }
            if (move.isEnpassant()) {
              ++result.enpassants;
            }
            if (move.getPromotedPiece()) {
              ++result.promotions;
            }
          }
        }
      }
    }
  }

public:
  using Board::Board;

  template <bool verbose>
  Result perft(uint32_t depth) {
    using namespace std::chrono;

    Result result{};

    auto start = high_resolution_clock::now();
    if (currentState().team_ == kWhite) {
      perftImpl<kWhite, verbose>(depth, result);
    } else {
      perftImpl<kBlack, verbose>(depth, result);
    }
    auto end = high_resolution_clock::now();
    
    auto ms = duration_cast<milliseconds>(end - start);
    uint64_t knps = (ms.count() > 0 ? static_cast<uint64_t>(static_cast<double>(result.nodes) / ms.count()) : result.nodes);
    std::cout << std::format("depth {}, nodes {}, time {}, speed {} knps\n", depth, result.nodes, ms, knps);

    if constexpr (verbose) {
      std::cout << std::format("captures {} enpassants {} castles {} promotions {}\n", result.captures, result.enpassants, result.castles, result.promotions);
    }
    return result;
  }
};
