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
  for (uint32_t i = 0; i < kSquareSize; ++letter) {
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

  // Update occupancy.
  for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
    boardState.occupancy_[kWhite] |= boardState.bitboards_[kWhite][piece];
    boardState.occupancy_[kBlack] |= boardState.bitboards_[kBlack][piece];
  }
  boardState.bothOccupancy_ = boardState.occupancy_[kWhite] | boardState.occupancy_[kBlack];

  // Parse team.
  std::string team; ss >> team;
  if (team == "w") {
    boardState.team_ = kWhite;
  } else {
    boardState.team_ = kBlack;
  }

  // Parse castle permisison.
  std::string castlePermission; ss >> castlePermission;
  if (castlePermission.find("K") != std::string::npos) {
    boardState.castlePermission_ |= kWhiteKingCastlePermission;
  }
  if (castlePermission.find("Q") != std::string::npos) {
    boardState.castlePermission_ |= kWhiteQueenCastlePermission;
  }
  if (castlePermission.find("k") != std::string::npos) {
    boardState.castlePermission_ |= kBlackKingCastlePermission;
  }
  if (castlePermission.find("q") != std::string::npos) {
    boardState.castlePermission_ |= kBlackQueenCastlePermission;
  }

  // Parse enpassant square.
  std::string enpassantSquare; ss >> enpassantSquare;
  boardState.enpassant_ = (enpassantSquare == "-" ? NO_SQUARE : stringToSquare(enpassantSquare));

  try {
    // Parse half move.
    std::string halfmove; ss >> halfmove;
    boardState.halfmove_ = static_cast<uint16_t>(std::stoi(halfmove));

    // Parse full move.
    std::string fullmove; ss >> fullmove;
    boardState.fullmove_ = static_cast<uint16_t>(std::stoi(fullmove));
  } catch (...) {
    // Most likely half move and full move are missing from the FEN.
  }
  return boardState;
}

std::ostream& operator<<(std::ostream& out, const BoardState& boardState) {
  using std::format;

  const auto findPieceAscii = [&](uint32_t square) {
    for (Team team : { kWhite, kBlack }) {
      for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
        if (isSquareSet(boardState.bitboards_[team][piece], square)) {
          return pieceToAsciiVisualOnly(team, piece);
        }
      }
    }
    return '.';
  };

  for (uint32_t i = 0; i < kSideSize; ++i) {
    out << format("{}|", kSideSize - i);
    for (uint32_t j = 0; j < kSideSize; ++j) {
      out << format(" {}", findPieceAscii(rankFileToSquare(i, j)));
    }
    out << '\n';
  }
  out << format("   a b c d e f g h\nTeam: {}\nCastle: {}\nEnpassant: {}\nhalfmove: {}\nfullmove: {}",
                teamToString(boardState.team_),
                castleToString(boardState.castlePermission_),
                squareToString(boardState.enpassant_),
                boardState.halfmove_,
                boardState.fullmove_);
  return out;
}

std::ostream& operator<<(std::ostream& out, const Board& board) {
  out << board.currentState_ << '\n';
  return out;
}
