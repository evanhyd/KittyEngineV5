#include "bitboard.h"
#include <cassert>
#include <format>
#include <iostream>

// Only used in printing out the UI
char pieceToAsciiVisualOnly(Team team, Piece piece) {
  constexpr std::array<std::array<char, kPieceSize>, kTeamSize> table = { {
    {'A', 'N', 'B', 'R', 'Q', 'K'},
    {'v', 'n', 'b', 'r', 'q', 'k'},
  } };
  return table[team][piece];
}

char pieceToAscii(Team team, Piece piece) {
  constexpr std::array<std::array<char, kPieceSize>, kTeamSize> table = { {
    {'P', 'N', 'B', 'R', 'Q', 'K'},
    {'p', 'n', 'b', 'r', 'q', 'k'},
  } };
  return table[team][piece];
}

std::pair<Team, Piece> asciiToPiece(char ascii) {
  for (Team team : {kWhite, kBlack}) {
    for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
      if (pieceToAscii(team, piece) == ascii) {
        return { team, piece };
      }
    }
  }
  assert(false && "invalid ascii value");
  return {}; // unreachable
}

std::string teamToString(Team team) {
  static const std::array<std::string, kTeamSize> table = {
    "white",
    "black",
  };
  return table[team];
}

std::string squareToString(Square square) {
  static const std::array<std::string, kSquareSize + 1> table = {
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "-",
  };
  return table[square];
}

Square stringToSquare(const std::string& squareString) {
  for (Square i = 0; i < kSquareSize; ++i) {
    if (squareToString(i) == squareString) {
      return i;
    }
  }
  assert(false && "invalid square");
  return NO_SQUARE;
}

std::string castleToString(Bitboard permission) {
  std::string str;
  if (permission & kWhiteKingCastlePermission) {
    str += 'K';
  }
  if (permission & kWhiteQueenCastlePermission) {
    str += 'Q';
  }
  if (permission & kBlackKingCastlePermission) {
    str += 'k';
  }
  if (permission & kBlackQueenCastlePermission) {
    str += 'q';
  }
  if (str.empty()) {
    str = "-";
  }
  return str;
}

void printBitboard(Bitboard bitboard) {
  using std::cout;
  using std::format;
  for (Square i = 0; i < kSideSize; ++i) {
    cout << format("{}|", kSideSize - i);
    for (Square j = 0; j < kSideSize; ++j) {
      cout << format(" {:d}", isSquareSet(bitboard, rankFileToSquare(i, j)));
    }
    cout << '\n';
  }
  cout << format("   a b c d e f g h\nBitboard Hex: {:#018x}\n\n", bitboard);
}
