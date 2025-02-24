#pragma once
#include "board.h"
#include <chrono>

class PerftDriver : public Board {
public:
  struct Result {
    uint64_t nodes;
    uint64_t captures;
    uint64_t enpassants;
    uint64_t castles;
    uint64_t promotions;
    uint64_t illegalMoves;
    uint64_t totalMoves;
  };

  using Board::Board;

  template <bool verbose>
  Result perft(uint32_t depth) {
    using namespace std::chrono;

    Result result{};
    auto start = high_resolution_clock::now();
    if (currentState_.color_ == kWhite) {
      perftImpl<kWhite, verbose>(depth, result);
    } else {
      perftImpl<kBlack, verbose>(depth, result);
    }
    auto end = high_resolution_clock::now();

    auto ms = duration_cast<milliseconds>(end - start);
    uint64_t knps = (ms.count() > 0 ? static_cast<uint64_t>(static_cast<double>(result.nodes) / ms.count()) : result.nodes);
    std::cout << std::format("depth {}, nodes {}, time {}, speed {} knps\n", depth, result.nodes, ms, knps);

    if constexpr (verbose) {
      std::cout << std::format("    captures {} enpassants {} castles {} promotions {} illegal moves {}/{} {:.3f}\n",
                               result.captures, result.enpassants, result.castles, result.promotions, 
                               result.illegalMoves, result.totalMoves, static_cast<double>(result.illegalMoves) / result.totalMoves);
    }
    return result;
  }

private:
  template <Color us, bool verbose>
  void perftImpl(uint32_t depth, Result& result) {
    constexpr Color defender = (us == kWhite ? kBlack : kWhite);
    const BoardState oldState = currentState_;
    MoveList moveList{};

    oldState.getPseudoMove<us>(moveList);
    for (Move move : moveList) {
      if (tryMakeMove<us>(move)) {
        if (depth > 1) {
          perftImpl<defender, verbose>(depth - 1, result);
        } else {
          ++result.nodes;
          if constexpr (verbose) {
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
      } else if constexpr (verbose) {
        ++result.illegalMoves;
      }

      if constexpr (verbose) {
        ++result.totalMoves;
      }

      currentState_ = oldState;
    }
  }
};
