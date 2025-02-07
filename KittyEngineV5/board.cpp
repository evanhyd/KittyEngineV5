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
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kWhiteKingCastle);
  }
  if (castlePermission.find("Q") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kWhiteQueenCastle);
  }
  if (castlePermission.find("k") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kBlackKingCastle);
  }
  if (castlePermission.find("q") != std::string::npos) {
    board.castlePermission_ = static_cast<CastlePermission>(board.castlePermission_ | kBlackQueenCastle);
  }

  // Parse enpassant square.
  std::string enpassantSquare; ss >> enpassantSquare;
  board.enpassant_ = (enpassantSquare == "-" ? NO_SQUARE : squareStringToSquare(enpassantSquare));

  // Parse half move.
  std::string halfmove; ss >> halfmove;
  board.halfmove_ = std::stoi(halfmove);

  // Parse full move.
  std::string fullmove; ss >> fullmove;
  board.fullmove_= std::stoi(fullmove);

  return board;
}

std::ostream& operator<<(std::ostream& out, const Board& board) {
  using std::format;

  const auto findPieceAscii = [&](uint32_t square) {
    for (Team team : { kWhite, kBlack }) {
      for (Piece piece : {kPawn, kKnight, kBishop, kRook, kQueen, kKing}) {
        if (isSquareSet(board.bitboards_[team][piece], square)) {
          return pieceToAscii(team, piece);
        }
      }
    }
    return '.';
  };

  for (uint32_t i = 0; i < kSideSize; ++i) {
    out << format("{}|", kSideSize - i);
    for (uint32_t j = 0; j < kSideSize; ++j) {
      out << format(" {}", findPieceAscii(i * kSideSize + j));
    }
    out << '\n';
  }
  out << format("   A B C D E F G H\nTeam: {}\nCastle: {}\nEnpassant: {}\nhalfmove: {}\nfullmove: {}", 
                teamToString(board.team_),
                castleToString(board.castlePermission_),
                squareToString(board.enpassant_),
                board.halfmove_,
                board.fullmove_);
  return out;
}
