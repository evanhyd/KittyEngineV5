#include "user_interface.h"
#include "boardstate.h"
#include <format>
#include <iostream>

namespace bb::user_interface {
  void UserInterface::printBoardState(std::ostream& out, const BoardState& boardState, Color sideToMove) {
    const auto findPieceAscii = [&](Square square) {
      for (Color color : {kWhite, kBlack}) {
        for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
          if (isSquareSet(boardState.bitboards_[color][piece], square)) {
            return notation::pieceToAsciiVisualOnly(color, piece);
          }
        }
      }
      return '.';
    };

    for (Square rank = 0; rank < kSideSize; ++rank) {
      out << std::format("{}|", kSideSize - rank);
      for (Square file = 0; file < kSideSize; ++file) {
        out << std::format(" {}", findPieceAscii(rankFileToSquare(rank, file)));
      }
      out << '\n';
    }
    out << std::format("   a b c d e f g h\nTeam: {}\nCastle: {}\nEnpassant: {}\nhalfmove: {}\nfullmove: {}",
                       notation::colorToString(sideToMove), notation::castleToString(boardState.castlePermission_),
                       notation::squareToString(boardState.enpassant_), boardState.halfmove_, boardState.fullmove_);
  }

  void printBitboard(Bitboard bitboard) {
    for (Square rank = 0; rank < kSideSize; ++rank) {
      std::cout << std::format("{}|", kSideSize - rank);
      for (Square file = 0; file < kSideSize; ++file) {
        std::cout << std::format(" {:d}", isSquareSet(bitboard, rankFileToSquare(rank, file)));
      }
      std::cout << '\n';
    }
    std::cout << std::format("   a b c d e f g h\nBitboard Hex: {:#018x}\n\n", bitboard);
  }
}
