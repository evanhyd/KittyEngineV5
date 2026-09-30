#include "boardstate.h"
#include "board.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_interface_policy.h"
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  struct TestInterfacePolicy {
    template <typename BoardType>
    void run(BoardType&) {}
    template <typename BoardType>
    void render(const BoardType&) {}
  };

  using TestBoard = Board<int, int, TestInterfacePolicy>;
}

static_assert(user_interface::UserInterfacePolicy<
  user_interface::TerminalInterfacePolicy, TestBoard>);

TEST(BoardState, DefaultConstructorZeroInitializesEveryField) {
  const BoardState state;
  for (Square square = 0; square < kSquareSize; ++square) {
    EXPECT_FALSE(state.getPieceAt(square).has_value());
  }
  EXPECT_EQ(state.getColorToMove(), 0u);
  EXPECT_EQ(state.getCastlingRights(), 0u);
  EXPECT_EQ(state.getEnpassantSquare(), 0u);
  EXPECT_EQ(state.getHalfmoveClock(), 0);
  EXPECT_EQ(state.getFullmoveNumber(), 0);
}

TEST(BoardState, ParsesFen) {
  const BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  EXPECT_EQ(state.getColorToMove(), kWhite);
  EXPECT_EQ(state.getPieceAt(E5), (std::tuple<Color, Piece>{kWhite, kPawn}));
  EXPECT_EQ(state.getPieceAt(D5), (std::tuple<Color, Piece>{kBlack, kPawn}));
  EXPECT_EQ(state.getEnpassantSquare(), D6);
  EXPECT_EQ(state.getCastlingRights(),
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kWhite] |
            kKingCastlePermission[kBlack] | kQueenCastlePermission[kBlack]);
  EXPECT_EQ(state.getHalfmoveClock(), 4);
  EXPECT_EQ(state.getFullmoveNumber(), 12);
}

TEST(BoardState, GettersReportPiecesAndPositionState) {
  const BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  ASSERT_TRUE(state.getPieceAt(E5).has_value());
  EXPECT_EQ(*state.getPieceAt(E5), (std::tuple<Color, Piece>{kWhite, kPawn}));
  EXPECT_EQ(*state.getPieceAt(D5), (std::tuple<Color, Piece>{kBlack, kPawn}));
  EXPECT_FALSE(state.getPieceAt(E4).has_value());
  EXPECT_THROW((void)state.getPieceAt(kNoSquare), std::out_of_range);
  EXPECT_EQ(state.getColorToMove(), kWhite);
  EXPECT_EQ(state.getCastlingRights(),
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kWhite] |
            kKingCastlePermission[kBlack] | kQueenCastlePermission[kBlack]);
  EXPECT_EQ(state.getEnpassantSquare(), D6);
  EXPECT_EQ(state.getHalfmoveClock(), 4);
  EXPECT_EQ(state.getFullmoveNumber(), 12);
}

TEST(BoardState, GettersTrackMakeAndUnmake) {
  BoardState state{fen::kStartPosition};
  const Move move{E2, E4, kPawn, kNoPiece, Move::kDoublePushFlag};
  const MoveUndo undo = state.makeMove<kWhite>(move);
  EXPECT_FALSE(state.getPieceAt(E2).has_value());
  EXPECT_EQ(*state.getPieceAt(E4), (std::tuple<Color, Piece>{kWhite, kPawn}));
  EXPECT_EQ(state.getColorToMove(), kBlack);
  EXPECT_EQ(state.getEnpassantSquare(), E3);
  EXPECT_EQ(state.getHalfmoveClock(), 0);
  EXPECT_EQ(state.getFullmoveNumber(), 1);
  state.unmakeMove<kWhite>(move, undo);
  EXPECT_EQ(*state.getPieceAt(E2), (std::tuple<Color, Piece>{kWhite, kPawn}));
  EXPECT_FALSE(state.getPieceAt(E4).has_value());
  EXPECT_EQ(state.getColorToMove(), kWhite);
  EXPECT_EQ(state.getEnpassantSquare(), kNoSquare);
  EXPECT_EQ(state.getFullmoveNumber(), 1);
}

TEST(Board, StartsAtInitialPosition) {
  const TestBoard board{0, 0, {}};
  EXPECT_EQ(board.getState().getColorToMove(), kWhite);
  EXPECT_EQ(*board.getState().getPieceAt(E1), (std::tuple<Color, Piece>{kWhite, kKing}));
  EXPECT_EQ(*board.getState().getPieceAt(E8), (std::tuple<Color, Piece>{kBlack, kKing}));
  EXPECT_EQ(board.getState().getFullmoveNumber(), 1);
}

TEST(BoardState, SetPositionReplacesAllFields) {
  BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  state.setPosition("7k/8/8/8/8/8/4p3/7K b - - 0 3");

  EXPECT_EQ(state.getColorToMove(), kBlack);
  EXPECT_FALSE(state.getPieceAt(E5).has_value());
  EXPECT_EQ(state.getPieceAt(E2), (std::tuple<Color, Piece>{kBlack, kPawn}));
  EXPECT_EQ(state.getCastlingRights(), 0u);
  EXPECT_EQ(state.getEnpassantSquare(), kNoSquare);
  EXPECT_EQ(state.getHalfmoveClock(), 0);
  EXPECT_EQ(state.getFullmoveNumber(), 3);
}

TEST(BoardState, InvalidSetPositionPreservesPreviousState) {
  BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  const BoardState original = state;
  EXPECT_THROW(state.setPosition("7k/8/8/8/8/8/4p3/7K b - - 0 bad"), std::invalid_argument);
  for (Square square = 0; square < kSquareSize; ++square) {
    EXPECT_EQ(state.getPieceAt(square), original.getPieceAt(square));
  }
  EXPECT_EQ(state.getColorToMove(), original.getColorToMove());
  EXPECT_EQ(state.getCastlingRights(), original.getCastlingRights());
  EXPECT_EQ(state.getEnpassantSquare(), original.getEnpassantSquare());
  EXPECT_EQ(state.getHalfmoveClock(), original.getHalfmoveClock());
  EXPECT_EQ(state.getFullmoveNumber(), original.getFullmoveNumber());
}

TEST(Notation, ParsesAndFormatsSharedChessNotation) {
  EXPECT_EQ(notation::asciiToPiece('P'), (std::pair<Color, Piece>{kWhite, kPawn}));
  EXPECT_EQ(notation::asciiToPiece('q'), (std::pair<Color, Piece>{kBlack, kQueen}));
  EXPECT_EQ(notation::stringToSquare("d6"), D6);
  EXPECT_EQ(notation::squareToString(D6), "d6");
  EXPECT_EQ(notation::stringToSquare("-"), kNoSquare);
  EXPECT_EQ(notation::stringToSide("b"), kBlack);
  EXPECT_EQ(notation::stringToCastling("Kq"),
            kKingCastlePermission[kWhite] | kQueenCastlePermission[kBlack]);
  EXPECT_EQ(notation::castleToString(notation::stringToCastling("Kq")), "Kq");
  EXPECT_THROW(notation::asciiToPiece('x'), std::invalid_argument);
  EXPECT_THROW(notation::stringToSquare("i9"), std::invalid_argument);
}

TEST(Notation, FormatsFullFen) {
  const BoardState start{fen::kStartPosition};
  EXPECT_EQ(notation::boardToFen(start), fen::kStartPosition);

  const BoardState custom{"r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq d6 4 12"};
  EXPECT_EQ(notation::boardToFen(custom),
            "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq d6 4 12");
}

TEST(TerminalInterfacePolicy, PrintsWelcomeMessage) {
  std::istringstream in;
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  user_interface::TerminalInterfacePolicy terminal(in, uciOut, humanOut);
  TestBoard board{0, 0, {}};
  terminal.run(board);
  EXPECT_NE(humanOut.str().find("Welcome to KittyEngineV5"), std::string::npos);
  EXPECT_NE(humanOut.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
  EXPECT_TRUE(uciOut.str().empty());
}

TEST(TerminalInterfacePolicy, RunStopsAtQuitOrEndOfInput) {
  std::istringstream in{"quit\n"};
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  user_interface::TerminalInterfacePolicy terminal(in, uciOut, humanOut);
  TestBoard board{0, 0, {}};
  terminal.run(board);
  EXPECT_EQ(humanOut.str().find("FEN: "), humanOut.str().rfind("FEN: "));
  EXPECT_TRUE(uciOut.str().empty());
}

TEST(TerminalInterfacePolicy, RendersBoardAndSuppressesUnchangedFrames) {
  std::istringstream in;
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  user_interface::TerminalInterfacePolicy terminal(in, uciOut, humanOut);
  TestBoard board{0, 0, {}};

  terminal.render(board);
  const std::string firstFrame = humanOut.str();
  EXPECT_NE(firstFrame.find(" | r | n | b | q | k | b | n | r | 8\n"), std::string::npos);
  EXPECT_NE(firstFrame.find(" | R | N | B | Q | K | B | N | R | 1\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("  -------------------------------\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("   a   b   c   d   e   f   g   h\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("Side to move: white\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("Castling: KQkq\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("En passant: -\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("Halfmove clock: 0\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("Fullmove number: 1\n"), std::string::npos);
  EXPECT_NE(firstFrame.find("FEN: " + std::string(fen::kStartPosition) + "\n"), std::string::npos);
  EXPECT_EQ(firstFrame.find('\x1b'), std::string::npos);

  terminal.render(board);
  EXPECT_EQ(humanOut.str(), firstFrame);

  board.getState().setPosition("7k/8/8/8/8/8/4p3/7K b - - 7 23");
  terminal.render(board);
  const std::string secondFrame = humanOut.str().substr(firstFrame.size());
  EXPECT_NE(secondFrame.find(" |   |   |   |   |   |   |   | k | 8\n"), std::string::npos);
  EXPECT_NE(secondFrame.find(" |   |   |   |   | v |   |   |   | 2\n"), std::string::npos);
  EXPECT_NE(secondFrame.find("Side to move: black\n"), std::string::npos);
  EXPECT_NE(secondFrame.find("Castling: -\n"), std::string::npos);
  EXPECT_NE(secondFrame.find("Halfmove clock: 7\n"), std::string::npos);
  EXPECT_NE(secondFrame.find("Fullmove number: 23\n"), std::string::npos);
  EXPECT_NE(secondFrame.find("FEN: 7k/8/8/8/8/8/4p3/7K b - - 7 23\n"), std::string::npos);
  EXPECT_EQ(secondFrame.find('\x1b'), std::string::npos);
  EXPECT_TRUE(uciOut.str().empty());
}

TEST(BoardState, RejectsMalformedFen) {
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8/8 x - -"}, std::invalid_argument);
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8 w - -"}, std::invalid_argument);
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8/8P w - -"}, std::invalid_argument);
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8/8 w - - 0"}, std::invalid_argument);
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8/8 w - - bad 1"}, std::invalid_argument);
  EXPECT_THROW(BoardState{"8/8/8/8/8/8/8/8 w - - 0 1 extra"}, std::invalid_argument);
}

