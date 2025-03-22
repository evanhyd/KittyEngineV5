#pragma once
#include "board.h"
#include <chrono>
#include <iostream>
#include "bitboard.h"
#include "move.h"
#include <cstdint>

namespace perft {
  struct Config {
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

  template <size_t depth>
  class PerftDriver {
  public:
    template <BoardStatus boardStatus, MoveType moveType>
    static constexpr void acceptMove(BoardState state, Move move) {
      if constexpr (depth <= 1) {
        ++internal::result.nodes;
      } else {
        constexpr Color our = boardStatus.color;
        constexpr Color their = getOtherColor(our);
        const Square srce = move.srce;
        const Square dest = move.dest;

        // Move the square.
        state.bitboards_[our][moveType.movedPiece] = moveSquare(state.bitboards_[our][moveType.movedPiece], srce, dest);
        state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], dest);
        state.bitboards_[their][kKnight] = unsetSquare(state.bitboards_[their][kKnight], dest);
        state.bitboards_[their][kBishop] = unsetSquare(state.bitboards_[their][kBishop], dest);
        state.bitboards_[their][kRook] = unsetSquare(state.bitboards_[their][kRook], dest);
        state.bitboards_[their][kQueen] = unsetSquare(state.bitboards_[their][kQueen], dest);

        // Reset enpassant square
        const Square enpassantSq = state.enpassant_;
        if constexpr (!moveType.isEnpassant && !moveType.isDoublePush) {
          state.enpassant_ = NO_SQUARE;
        }

        // Update castle occupancy.
        if constexpr (boardStatus.kingCastle[kWhite] || boardStatus.kingCastle[kBlack] || boardStatus.queenCastle[kWhite] || boardStatus.queenCastle[kBlack]) {
          state.castlePermission_ = unsetSquare(unsetSquare(state.castlePermission_, srce), dest);
        }

        // Update the color.
        state.color_ = their;

        // Update half move and full move.
        ++state.halfmove_;
        /*if (move.isCaptured()) {
          state.halfmove_ = 0;
        }*/
        if constexpr (our == kBlack) {
          ++state.fullmove_;
        }
        

        if constexpr (moveType.movedPiece == kPawn) {
          state.halfmove_ = 0;

          if constexpr (moveType.isEnpassant) {
            state.bitboards_[their][kPawn] = unsetSquare(state.bitboards_[their][kPawn], (their == kWhite ? 
                                                                                          squareUp(enpassantSq) : 
                                                                                          squareDown(enpassantSq)));
            state.enumerateMoves<boardStatus.normalMove(), PerftDriver<depth - 1>>();

          } else if constexpr (moveType.isDoublePush) {
            state.enpassant_ = (our == kWhite ? squareUp(srce) : squareDown(srce));
            state.enumerateMoves<boardStatus.doublePushMove(), PerftDriver<depth - 1>>();

          } else if constexpr (moveType.promotionPiece) {
            state.bitboards_[our][kPawn] = unsetSquare(state.bitboards_[our][kPawn], dest);
            state.bitboards_[our][moveType.promotionPiece] = setSquare(state.bitboards_[our][moveType.promotionPiece], dest);
            state.enumerateMoves<boardStatus.normalMove(), PerftDriver<depth - 1>>();

          } else {
            state.enumerateMoves<boardStatus.normalMove(), PerftDriver<depth - 1>>();
          }

        } else if constexpr (moveType.movedPiece == kKing) {
          if constexpr (moveType.isKingSideCastle) {
            state.bitboards_[our][kRook] = (our == kWhite ?
                                            moveSquare(state.bitboards_[our][kRook], H1, F1) :
                                            moveSquare(state.bitboards_[our][kRook], H8, F8));
            state.enumerateMoves<boardStatus.kingCastleMove(), PerftDriver<depth - 1>>();

          } else if constexpr (moveType.isQueenSideCastle) {
            state.bitboards_[our][kRook] = (our == kWhite ?
                                            moveSquare(state.bitboards_[our][kRook], A1, D1) :
                                            moveSquare(state.bitboards_[our][kRook], A8, D8));
            state.enumerateMoves<boardStatus.queenCastleMove(), PerftDriver<depth - 1>>();
          } else {
            state.enumerateMoves<boardStatus.normalKingMove(), PerftDriver<depth - 1>>();
          }
        } else {
          state.enumerateMoves<boardStatus.normalMove(), PerftDriver<depth - 1>>();
        }
      }
    }
  };

  namespace internal {
    template <size_t depth>
    inline constexpr void runPerftWithState(const BoardState& state) {
      const bool color = state.color_;
      const bool hasEnpassant = (state.enpassant_ != NO_SQUARE);
      const bool whiteKingCastle = (state.castlePermission_ & kKingCastlePermission[kWhite]) == kKingCastlePermission[kWhite];
      const bool blackKingCastle = (state.castlePermission_ & kKingCastlePermission[kBlack]) == kKingCastlePermission[kBlack];
      const bool whiteQueenCastle = (state.castlePermission_ & kQueenCastlePermission[kWhite]) == kQueenCastlePermission[kWhite];
      const bool blackQueenCastle = (state.castlePermission_ & kQueenCastlePermission[kBlack]) == kQueenCastlePermission[kBlack];

      if (color == kWhite && !hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {false, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && !hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, false, {true, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {false, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kWhite && hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kWhite, true, {true, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {false, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && !hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, false, {true, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && !whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {false, true}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, false}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && !blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, false}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, false}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && !blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, false}, {true, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, true}, {false, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && blackKingCastle && !whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, true}, {false, true} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && !blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, true}, {true, false} }, PerftDriver<depth >> ();
      } else if (color == kBlack && hasEnpassant && whiteKingCastle && blackKingCastle && whiteQueenCastle && blackQueenCastle) {
        state.enumerateMoves < BoardStatus{ kBlack, true, {true, true}, {true, true} }, PerftDriver<depth >> ();
      }
     }
  }

  template <Config config, bool canPrint = true>
  inline constexpr Result runPerft(const BoardState& state, uint32_t depth) {
    using namespace std::chrono;

    internal::result = Result{};

    auto start = high_resolution_clock::now();
    switch (depth) {
    case 1: internal::runPerftWithState<1>(state); break;
    case 2: internal::runPerftWithState<2>(state); break;
    case 3: internal::runPerftWithState<3>(state); break;
    case 4: internal::runPerftWithState<4>(state); break;
    case 5: internal::runPerftWithState<5>(state); break;
    case 6: internal::runPerftWithState<6>(state); break;
    case 7: internal::runPerftWithState<7>(state); break;
    case 8: internal::runPerftWithState<8>(state); break;
    case 9: internal::runPerftWithState<9>(state); break;
    case 10: internal::runPerftWithState<10>(state); break;
    case 11: internal::runPerftWithState<11>(state); break;
    case 12: internal::runPerftWithState<12>(state); break;
    case 13: internal::runPerftWithState<13>(state); break;
    case 14: internal::runPerftWithState<14>(state); break;
    case 15: internal::runPerftWithState<15>(state); break;
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
