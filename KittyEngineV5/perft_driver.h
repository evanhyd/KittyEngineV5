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

  template <Color our, size_t depth>
  class PerftDriver {
  public:
    static constexpr void acceptMove(BoardState state, Move move) {
      if constexpr (depth <= 1) {
        ++internal::result.nodes;
      } else {
        constexpr Color their = getOtherColor(our);
        const Square srce = move.getSourceSquare();
        const Square dest = move.getDestinationSquare();
        const Piece movedPiece = move.getMovedPiece();

        // Move the square.
        state.bitboards_[our][movedPiece] = moveSquare(state.bitboards_[our][movedPiece], srce, dest);
        state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], dest);
        state.bitboards_[their][kKnight] = unsetSquare(state.bitboards_[their][kKnight], dest);
        state.bitboards_[their][kBishop] = unsetSquare(state.bitboards_[their][kBishop], dest);
        state.bitboards_[their][kRook] = unsetSquare(state.bitboards_[their][kRook], dest);
        state.bitboards_[their][kQueen] = unsetSquare(state.bitboards_[their][kQueen], dest);

        // Reset enpassant square
        const Square enpassantSq = state.enpassant_;
        state.enpassant_ = NO_SQUARE;

        // Update castle occupancy.
        state.castlePermission_ = unsetSquare(unsetSquare(state.castlePermission_, srce), dest);

        // Update half move and full move.
        ++state.halfmove_;
        /*if (move.isCaptured()) {
          state.halfmove_ = 0;
        }*/

        if constexpr (our == kBlack) {
          ++state.fullmove_;
        }

        if (movedPiece == kPawn) {
          state.halfmove_ = 0;

          if (move.isEnpassant()) {
            if constexpr (their == kWhite) {
              state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], squareUp(enpassantSq));
            } else {
              state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], squareDown(enpassantSq));
            }
          } else if (move.isDoublePush()) {
            if constexpr (our == kWhite) {
              state.enpassant_ = squareUp(srce);
            } else {
              state.enpassant_ = squareDown(srce);
            }
          } else if (Piece promotedPiece = move.getPromotedPiece(); promotedPiece) {
            state.bitboards_[our][kPawn] = unsetSquare(state.bitboards_[our][kPawn], dest);
            state.bitboards_[our][promotedPiece] = setSquare(state.bitboards_[our][promotedPiece], dest);
          }

        } else if (movedPiece == kKing) {
          if (move.isKingSideCastle()) {
            if constexpr (our == kWhite) {
              state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], H1, F1);
            } else {
              state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], H8, F8);
            }
          } else if (move.isQueenSideCastle()) {
            if constexpr (our == kWhite) {
              state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], A1, D1);
            } else {
              state.bitboards_[our][kRook] = moveSquare(state.bitboards_[our][kRook], A8, D8);
            }
          }
        }

        state.color_ = getOtherColor(our);
        state.enumerateMoves<their, PerftDriver<their, depth - 1>>();
      }
    }

  };

  template <Config config, bool canPrint = true>
  inline constexpr Result runPerft(const BoardState& state, uint32_t depth) {
    static_assert(!(config.isBulkCount && config.isDetailed), "bulk counting is incompatiable with detailed perft");
    using namespace std::chrono;

    internal::result = Result{};

    auto start = high_resolution_clock::now();
    switch (depth) {
    case 1: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 1>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 1>>(); break;
    case 2: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 2>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 2>>(); break;
    case 3: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 3>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 3>>(); break;
    case 4: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 4>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 4>>(); break;
    case 5: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 5>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 5>>(); break;
    case 6: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 6>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 6>>(); break;
    case 7: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 7>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 7>>(); break;
    case 8: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 8>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 8>>(); break;
    case 9: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 9>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 9>>(); break;
    case 10: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 10>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 10>>(); break;
    case 11: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 11>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 11>>(); break;
    case 12: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 12>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 12>>(); break;
    case 13: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 13>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 13>>(); break;
    case 14: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 14>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 14>>(); break;
    case 15: state.getColor() == kWhite ? state.enumerateMoves<kWhite, PerftDriver<kWhite, 15>>() : state.enumerateMoves<kBlack, PerftDriver<kBlack, 15>>(); break;
    default: break;
    }
    auto end = high_resolution_clock::now();

    if constexpr (canPrint) {
      auto ms = duration_cast<milliseconds>(end - start);
      uint64_t knps = (ms.count() > 0 ? static_cast<uint64_t>(static_cast<double>(internal::result.nodes) / ms.count()) : internal::result.nodes);
      std::cout << std::format("depth {}, nodes {}, time {}, speed {} knps\n", depth, internal::result.nodes, ms, knps);
      if constexpr (config.isDetailed) {
        std::cout << std::format("    captures {} enpassants {} castles {} promotions {}\n",
                                 internal::result.captures, internal::result.enpassants, internal::result.castles, internal::result.promotions);
      }
    }

    return internal::result;
  }
}
