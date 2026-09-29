#include "boardstate.h"
#include "notation.h"
#include "user_interface.h"
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>

using namespace bb;

TEST(BoardState, ParsesFen) {
  const auto [state, sideToMove] = BoardState::fromFEN("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12");
  EXPECT_EQ(sideToMove, kWhite);
  EXPECT_TRUE(isSquareSet(state.bitboards_[kWhite][kPawn], E5));
  EXPECT_TRUE(isSquareSet(state.bitboards_[kBlack][kPawn], D5));
  EXPECT_EQ(state.enpassant_, D6);
  EXPECT_EQ(state.castlePermission_,
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kWhite] |
            kKingCastlePermission[kBlack] | kQueenCastlePermission[kBlack]);
  EXPECT_EQ(state.halfmove_, 4u);
  EXPECT_EQ(state.fullmove_, 12u);
}

TEST(Notation, ParsesAndFormatsSharedChessNotation) {
  EXPECT_EQ(notation::asciiToPiece('P'), (std::pair<Color, Piece>{kWhite, kPawn}));
  EXPECT_EQ(notation::asciiToPiece('q'), (std::pair<Color, Piece>{kBlack, kQueen}));
  EXPECT_EQ(notation::stringToSquare("d6"), D6);
  EXPECT_EQ(notation::squareToString(D6), "d6");
  EXPECT_EQ(notation::stringToSquare("-"), NO_SQUARE);
  EXPECT_EQ(notation::parseSideToMove("b"), kBlack);
  EXPECT_EQ(notation::parseCastlingRights("Kq"),
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kBlack]);
  EXPECT_THROW(notation::asciiToPiece('x'), std::invalid_argument);
  EXPECT_THROW(notation::stringToSquare("i9"), std::invalid_argument);
}

TEST(UserInterface, PrintsBoardStateAndExternalSideToMove) {
  // source: https://www.chessprogramming.org/Perft_Results#initial-position
  const auto [state, sideToMove] = BoardState::fromFEN(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
  std::ostringstream out;
  user_interface::UserInterface::printBoardState(out, state, sideToMove);
  EXPECT_NE(out.str().find("Team: black"), std::string::npos);
  EXPECT_NE(out.str().find("Castle: KQkq"), std::string::npos);
}

TEST(BoardState, RejectsMalformedFen) {
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8/8 x - -"), std::invalid_argument);
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8 w - -"), std::invalid_argument);
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8/8P w - -"), std::invalid_argument);
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8/8 w - - 0"), std::invalid_argument);
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8/8 w - - bad 1"), std::invalid_argument);
  EXPECT_THROW(BoardState::fromFEN("8/8/8/8/8/8/8/8 w - - 0 1 extra"), std::invalid_argument);
}

