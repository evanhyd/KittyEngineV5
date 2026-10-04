#include "notation.h"
#include "boardstate.h"
#include <stdexcept>

namespace bb::notation {
  char pieceToAsciiVisual(Side side, Piece piece) {
    static constexpr std::array<std::array<char, kPieceSize>, kSideSize> table{{
      {'A', 'N', 'B', 'R', 'Q', 'K'},
      {'v', 'n', 'b', 'r', 'q', 'k'},
    }};
    return table[side][piece];
  }

  char pieceToAscii(Side side, Piece piece) {
    static constexpr std::array<std::array<char, kPieceSize>, kSideSize> table{{
      {'P', 'N', 'B', 'R', 'Q', 'K'},
      {'p', 'n', 'b', 'r', 'q', 'k'},
    }};
    return table[side][piece];
  }

  std::pair<Side, Piece> asciiToPiece(char ascii) {
    const Side side = ascii >= 'A' && ascii <= 'Z' ? White : Black;
    const char lower = side == White ? static_cast<char>(ascii - 'A' + 'a') : ascii;
    switch (lower) {
    case 'p':
      return {side, Pawn};
    case 'n':
      return {side, Knight};
    case 'b':
      return {side, Bishop};
    case 'r':
      return {side, Rook};
    case 'q':
      return {side, Queen};
    case 'k':
      return {side, King};
    default:
      throw std::invalid_argument("Invalid piece letter");
    }
  }

  Side stringToSide(std::string_view side) {
    if (side == "w") {
      return White;
    }
    if (side == "b") {
      return Black;
    }
    throw std::invalid_argument("Invalid FEN side to move");
  }

  std::string sideToString(Side side) {
    if (side == White) {
      return "white";
    }
    if (side == Black) {
      return "black";
    }
    throw std::invalid_argument("Invalid side");
  }

  Square stringToSquare(std::string_view square) {
    if (square == "-") {
      return NoSquare;
    }
    if (square.size() != 2 || square[0] < 'a' || square[0] > 'h' ||
        square[1] < '1' || square[1] > '8') {
      throw std::invalid_argument("Invalid square notation");
    }
    return rankFileToSquare('8' - square[1], square[0] - 'a');
  }

  std::string squareToString(Square square) {
    if (square == NoSquare) {
      return "-";
    }
    if (square >= kSquareSize) {
      throw std::invalid_argument("Invalid square");
    }
    return {static_cast<char>('a' + getSquareFile(square)),
            static_cast<char>('8' - getSquareRank(square))};
  }


  Bitboard stringToCastling(std::string_view rights) {
    if (rights == "-") {
      return 0;
    }
    Bitboard permission = 0;
    for (const char right : rights) {
      switch (right) {
      case 'K':
        permission |= kKingCastlePermission[White];
        break;
      case 'Q':
        permission |= kQueenCastlePermission[White];
        break;
      case 'k':
        permission |= kKingCastlePermission[Black];
        break;
      case 'q':
        permission |= kQueenCastlePermission[Black];
        break;
      default:
        throw std::invalid_argument("Invalid FEN castling rights");
      }
    }
    return permission;
  }

  std::string castleToString(Bitboard permission) {
    std::string result;
    if ((permission & kKingCastlePermission[White]) == kKingCastlePermission[White]) {
      result += 'K';
    }
    if ((permission & kQueenCastlePermission[White]) == kQueenCastlePermission[White]) {
      result += 'Q';
    }
    if ((permission & kKingCastlePermission[Black]) == kKingCastlePermission[Black]) {
      result += 'k';
    }
    if ((permission & kQueenCastlePermission[Black]) == kQueenCastlePermission[Black]) {
      result += 'q';
    }
    if (result.empty()) {
      return "-";
    }
    return result;
  }

  std::string boardToFen(const BoardState& state) {
    std::string placement;
    for (Square rank = 0; rank < kBoardLenSize; ++rank) {
      if (rank != 0) {
        placement += '/';
      }
      int emptySquares = 0;
      for (Square file = 0; file < kBoardLenSize; ++file) {
        const auto occupant = state.getPieceAt(rankFileToSquare(rank, file));
        if (!occupant) {
          ++emptySquares;
          continue;
        }
        if (emptySquares != 0) {
          placement += static_cast<char>('0' + emptySquares);
          emptySquares = 0;
        }
        placement += pieceToAscii(std::get<0>(*occupant), std::get<1>(*occupant));
      }
      if (emptySquares != 0) {
        placement += static_cast<char>('0' + emptySquares);
      }
    }

    const Side side = state.getSideToMove();
    if (side != White && side != Black) {
      throw std::invalid_argument("Invalid side to move");
    }
    placement += side == White ? " w " : " b ";
    placement += castleToString(state.getCastlingRights());
    placement += ' ';
    placement += squareToString(state.getEnpassantSquare());
    placement += ' ';
    placement += std::to_string(state.getHalfmoveClock());
    placement += ' ';
    placement += std::to_string(state.getFullmoveNumber());
    return placement;
  }

  //Move stringToMove(std::string_view moveNotation) {
  //  if (moveNotation.size() < 4) {
  //    throw std::invalid_argument("Invalid move notation");
  //  }
  //  Square srce = stringToSquare(moveNotation.substr(0, 2));
  //  Square dest = stringToSquare(moveNotation.substr(2, 2));
  //  Piece promotedPiece = [&]() -> Piece {
  //    if (moveNotation.size() == 5) {
  //      return asciiToPiece(moveNotation[4]).second;
  //    }
  //    return kNoPiece;
  //  }();
  //}

  std::string moveToString(Move move) {
    std::string str = squareToString(move.getSource()) + squareToString(move.getDest());
    if (Piece promotedPiece = move.getPromotedPieceType(); promotedPiece != NoPiece) {
      str += pieceToAscii(Black, promotedPiece);
    }
    return str;
  }
}
