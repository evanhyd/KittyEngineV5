#pragma once
#include "board.h"
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

  namespace internal {
    inline Result result{};
  }

  template <size_t depth, bool detailed>
  class PerftDriver {
  public:
    template <MoveType moveType>
    static constexpr void acceptMove(const BoardState& state, Move<moveType> move) {
      if constexpr (depth <= 1) {
        ++internal::result.nodes;
        if constexpr (detailed) {
          constexpr Color their = getOtherColor(moveType.color);
          const Bitboard target = toBitboard(move.dest);
          if (moveType.isEnpassant ||
              (target & (state.bitboards_[their][kPawn] | state.bitboards_[their][kKnight] |
                         state.bitboards_[their][kBishop] | state.bitboards_[their][kRook] |
                         state.bitboards_[their][kQueen]))) {
            ++internal::result.captures;
          }
          internal::result.enpassants += moveType.isEnpassant;
          internal::result.castles += moveType.isKingSideCastle || moveType.isQueenSideCastle;
          internal::result.promotions += moveType.promotionPiece != 0;
        }
      } else {
        const BoardState child = state.makeMove<moveType>(move);
        child.enumerateMoves<getOtherColor(moveType.color), PerftDriver<depth-1, detailed>>();
      }
    }

  };

  template <Config config>
  inline Result countPerft(const BoardState& state, uint32_t depth) {
    static_assert(!(config.isBulkCount && config.isDetailed), "bulk counting is incompatiable with detailed perft");
    internal::result = Result{};
    switch (depth) {
    case 1: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<1, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<1, config.isDetailed>>(); break;
    case 2: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<2, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<2, config.isDetailed>>(); break;
    case 3: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<3, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<3, config.isDetailed>>(); break;
    case 4: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<4, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<4, config.isDetailed>>(); break;
    case 5: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<5, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<5, config.isDetailed>>(); break;
    case 6: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<6, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<6, config.isDetailed>>(); break;
    case 7: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<7, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<7, config.isDetailed>>(); break;
    case 8: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<8, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<8, config.isDetailed>>(); break;
    case 9: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<9, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<9, config.isDetailed>>(); break;
    case 10: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<10, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<10, config.isDetailed>>(); break;
    case 11: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<11, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<11, config.isDetailed>>(); break;
    case 12: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<12, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<12, config.isDetailed>>(); break;
    case 13: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<13, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<13, config.isDetailed>>(); break;
    case 14: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<14, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<14, config.isDetailed>>(); break;
    case 15: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<15, config.isDetailed>>() : state.enumerateMoves<kBlack, PerftDriver<15, config.isDetailed>>(); break;
    default: break;
    }
    return internal::result;
  }

  template <Config config, bool canPrint = true>
  inline Result runPerft(const BoardState& state, uint32_t depth) {
    using namespace std::chrono;
    const auto start = high_resolution_clock::now();
    const Result result = countPerft<config>(state, depth);
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
}
