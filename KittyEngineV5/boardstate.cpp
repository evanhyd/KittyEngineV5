#include "board.h"
#include <format>
#include <iostream>
#include <sstream>

BoardState BoardState::fromFEN(const std::string& fen) {
  BoardState boardState{};
  std::istringstream ss(fen);

  // Parse positions.
  std::string position; ss >> position;
  auto letter = position.begin();
  for (Square i = 0; i < kSquareSize; ++letter) {
    if (isdigit(*letter)) {
      // Skip empty square.
      i += *letter - '0';
    } else if (isalpha(*letter)) {
      // Must be piece
      auto [team, piece] = asciiToPiece(*letter);
      boardState.bitboards_[team][piece] = setSquare(boardState.bitboards_[team][piece], i);
      ++i;
    } // else Ignore rank separator.
  }

  // Parse team.
  std::string team; ss >> team;
  if (team == "w") {
    boardState.color_ = kWhite;
  } else {
    boardState.color_ = kBlack;
  }

  // Parse castle permisison.
  std::string castlePermission; ss >> castlePermission;
  if (castlePermission.find("K") != std::string::npos) {
    boardState.castlePermission_ |= kKingCastlePermission[kWhite];
  }
  if (castlePermission.find("Q") != std::string::npos) {
    boardState.castlePermission_ |= kQueenCastlePermission[kWhite];
  }
  if (castlePermission.find("k") != std::string::npos) {
    boardState.castlePermission_ |= kKingCastlePermission[kBlack];
  }
  if (castlePermission.find("q") != std::string::npos) {
    boardState.castlePermission_ |= kQueenCastlePermission[kBlack];
  }

  // Parse enpassant square.
  std::string enpassantSquare; ss >> enpassantSquare;
  boardState.enpassant_ = (enpassantSquare == "-" ? NO_SQUARE : stringToSquare(enpassantSquare));

  // Parse half move and full move.
  if (ss >> boardState.halfmove_) {
    ss >> boardState.fullmove_;
  }
  return boardState;
}

std::ostream& operator<<(std::ostream& out, const BoardState& boardState) {
  using std::format;

  const auto findPieceAscii = [&](Square square) {
    for (Color team : { kWhite, kBlack }) {
      for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
        if (isSquareSet(boardState.bitboards_[team][piece], square)) {
          return pieceToAsciiVisualOnly(team, piece);
        }
      }
    }
    return '.';
  };

  for (Square i = 0; i < kSideSize; ++i) {
    out << format("{}|", kSideSize - i);
    for (Square j = 0; j < kSideSize; ++j) {
      out << format(" {}", findPieceAscii(rankFileToSquare(i, j)));
    }
    out << '\n';
  }
  out << format("   a b c d e f g h\nTeam: {}\nCastle: {}\nEnpassant: {}\nhalfmove: {}\nfullmove: {}",
                colorToString(boardState.color_),
                castleToString(boardState.castlePermission_),
                squareToString(boardState.enpassant_),
                boardState.halfmove_,
                boardState.fullmove_);
  return out;
}
