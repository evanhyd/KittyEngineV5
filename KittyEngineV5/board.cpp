#include "board.h"
#include <format>
#include <iostream>
#include <sstream>

Board Board::fromFEN(const std::string& fen) {
  Board board{};
  std::istringstream ss(fen);

  // Parse positions.
  std::string position; ss >> position;
  auto letter = position.begin();
  for (size_t i = 0; i < kSquareSize; ++letter) {
    if (isdigit(*letter)) {
      // Skip empty square.
      i += *letter - '0';
    } else if (isalpha(*letter)) {
      // Must be piece
      auto [team, piece] = asciiToPiece(*letter);
      board.bitboards_[team][piece] = setSquare(board.bitboards_[team][piece], i);
      ++i;
    } // else Ignore rank separator.
  }

  // Update occupancy.
  for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
    board.occupancy_[kWhite] |= board.bitboards_[kWhite][piece];
    board.occupancy_[kBlack] |= board.bitboards_[kBlack][piece];
  }
  board.bothOccupancy_ = board.occupancy_[kWhite] | board.occupancy_[kBlack];

  // Parse team.
  std::string team; ss >> team;
  if (team == "w") {
    board.team_ = kWhite;
  } else {
    board.team_ = kBlack;
  }

  // Parse castle permisison.
  std::string castlePermission; ss >> castlePermission;
  if (castlePermission.find("K") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kWhiteKingCastlePermission);
  }
  if (castlePermission.find("Q") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kWhiteQueenCastlePermission);
  }
  if (castlePermission.find("k") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kBlackKingCastlePermission);
  }
  if (castlePermission.find("q") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kBlackQueenCastlePermission);
  }

  // Parse enpassant square.
  std::string enpassantSquare; ss >> enpassantSquare;
  board.enpassant_ = (enpassantSquare == "-" ? NO_SQUARE : stringToSquare(enpassantSquare));

  try {
    // Parse half move.
    std::string halfmove; ss >> halfmove;
    board.halfmove_ = std::stoi(halfmove);

    // Parse full move.
    std::string fullmove; ss >> fullmove;
    board.fullmove_ = std::stoi(fullmove);
  } catch (...) {
    // Most likely half move and full move are missing from the FEN.
  }
  return board;
}

std::ostream& operator<<(std::ostream& out, const Board& board) {
  using std::format;

  const auto findPieceAscii = [&](uint32_t square) {
    for (Team team : { kWhite, kBlack }) {
      for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
        if (isSquareSet(board.bitboards_[team][piece], square)) {
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
                teamToString(board.team_),
                castleToString(board.castlePermission_),
                squareToString(board.enpassant_),
                board.halfmove_,
                board.fullmove_);
  return out;
}
