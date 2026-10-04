#include "boardstate.h"
#include "board.h"
#include "handcraft_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_ui.h"
#include <sstream>
#include <stdexcept>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  using TestSearch = searching::NegamaxSearchPolicy<evaluation::HandCraftEvaluationPolicy>;
  using TestBoard = Board<TestSearch>;
}

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
  EXPECT_EQ(state.getColorToMove(), White);
  EXPECT_EQ(state.getPieceAt(E5), (std::tuple<Color, Piece>{White, Pawn}));
  EXPECT_EQ(state.getPieceAt(D5), (std::tuple<Color, Piece>{Black, Pawn}));
  EXPECT_EQ(state.getEnpassantSquare(), D6);
  EXPECT_EQ(state.getCastlingRights(),
            kKingCastlePermission[White] | kQueenCastlePermission[White] |
            kKingCastlePermission[Black] | kQueenCastlePermission[Black]);
  EXPECT_EQ(state.getHalfmoveClock(), 4);
  EXPECT_EQ(state.getFullmoveNumber(), 12);
}

TEST(BoardState, GettersReportPiecesAndPositionState) {
  const BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  ASSERT_TRUE(state.getPieceAt(E5).has_value());
  EXPECT_EQ(*state.getPieceAt(E5), (std::tuple<Color, Piece>{White, Pawn}));
  EXPECT_EQ(*state.getPieceAt(D5), (std::tuple<Color, Piece>{Black, Pawn}));
  EXPECT_FALSE(state.getPieceAt(E4).has_value());
  EXPECT_THROW((void)state.getPieceAt(NoSquare), std::out_of_range);
  EXPECT_EQ(state.getColorToMove(), White);
  EXPECT_EQ(state.getCastlingRights(),
            kKingCastlePermission[White] | kQueenCastlePermission[White] |
            kKingCastlePermission[Black] | kQueenCastlePermission[Black]);
  EXPECT_EQ(state.getEnpassantSquare(), D6);
  EXPECT_EQ(state.getHalfmoveClock(), 4);
  EXPECT_EQ(state.getFullmoveNumber(), 12);
}

TEST(BoardState, GettersTrackMakeAndUnmake) {
  BoardState state{fen::kStartPosition};
  const Move move{E2, E4, Pawn, NoPiece, Move::kDoublePushFlag};
  const MoveUndo undo = state.makeMove<White>(move);
  EXPECT_FALSE(state.getPieceAt(E2).has_value());
  EXPECT_EQ(*state.getPieceAt(E4), (std::tuple<Color, Piece>{White, Pawn}));
  EXPECT_EQ(state.getColorToMove(), Black);
  EXPECT_EQ(state.getEnpassantSquare(), E3);
  EXPECT_EQ(state.getHalfmoveClock(), 0);
  EXPECT_EQ(state.getFullmoveNumber(), 1);
  state.unmakeMove<White>(move, undo);
  EXPECT_EQ(*state.getPieceAt(E2), (std::tuple<Color, Piece>{White, Pawn}));
  EXPECT_FALSE(state.getPieceAt(E4).has_value());
  EXPECT_EQ(state.getColorToMove(), White);
  EXPECT_EQ(state.getEnpassantSquare(), NoSquare);
  EXPECT_EQ(state.getFullmoveNumber(), 1);
}

TEST(Board, StartsAtInitialPosition) {
  const TestBoard board{TestSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  EXPECT_EQ(board.getState().getColorToMove(), White);
  EXPECT_EQ(*board.getState().getPieceAt(E1), (std::tuple<Color, Piece>{White, King}));
  EXPECT_EQ(*board.getState().getPieceAt(E8), (std::tuple<Color, Piece>{Black, King}));
  EXPECT_EQ(board.getState().getFullmoveNumber(), 1);
}

TEST(Board, RetainsStatefulEvaluatorAcrossSearches) {
  struct StatefulEvaluator {
    int* calls;
    int* copies;
    const void** firstAddress;
    bool* stableAddress;

    StatefulEvaluator(int& calls, int& copies, const void*& firstAddress, bool& stableAddress)
      : calls(&calls), copies(&copies), firstAddress(&firstAddress),
        stableAddress(&stableAddress) {}

    StatefulEvaluator(const StatefulEvaluator& other)
      : calls(other.calls), copies(other.copies), firstAddress(other.firstAddress),
        stableAddress(other.stableAddress) {
      ++*copies;
    }

    int32_t evaluate(const BoardState&) {
      if (*firstAddress == nullptr) {
        *firstAddress = this;
      } else if (*firstAddress != this) {
        *stableAddress = false;
      }
      return ++*calls;
    }
  };

  int calls = 0;
  int copies = 0;
  const void* firstAddress = nullptr;
  bool stableAddress = true;
  Board board{searching::NegamaxSearchPolicy{
    StatefulEvaluator{calls, copies, firstAddress, stableAddress}, 50}};
  const int copiesBeforeSearch = copies;

  ASSERT_TRUE(board.search(1, [](const searching::SearchResult&) {}).bestMove.has_value());
  const int callsAfterFirstSearch = calls;
  ASSERT_TRUE(board.search(1, [](const searching::SearchResult&) {}).bestMove.has_value());

  EXPECT_GT(callsAfterFirstSearch, 0);
  EXPECT_GT(calls, callsAfterFirstSearch);
  EXPECT_NE(firstAddress, nullptr);
  EXPECT_TRUE(stableAddress);
  EXPECT_EQ(copies, copiesBeforeSearch);
}

TEST(BoardState, SetPositionReplacesAllFields) {
  BoardState state{"r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 4 12"};
  state.setPosition("7k/8/8/8/8/8/4p3/7K b - - 0 3");

  EXPECT_EQ(state.getColorToMove(), Black);
  EXPECT_FALSE(state.getPieceAt(E5).has_value());
  EXPECT_EQ(state.getPieceAt(E2), (std::tuple<Color, Piece>{Black, Pawn}));
  EXPECT_EQ(state.getCastlingRights(), 0u);
  EXPECT_EQ(state.getEnpassantSquare(), NoSquare);
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
  EXPECT_EQ(notation::asciiToPiece('P'), (std::pair<Color, Piece>{White, Pawn}));
  EXPECT_EQ(notation::asciiToPiece('q'), (std::pair<Color, Piece>{Black, Queen}));
  EXPECT_EQ(notation::stringToSquare("d6"), D6);
  EXPECT_EQ(notation::squareToString(D6), "d6");
  EXPECT_EQ(notation::stringToSquare("-"), NoSquare);
  EXPECT_EQ(notation::stringToSide("b"), Black);
  EXPECT_EQ(notation::stringToCastling("Kq"),
            kKingCastlePermission[White] | kQueenCastlePermission[Black]);
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

TEST(TerminalUI, PrintsWelcomeMessage) {
  std::istringstream in;
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  TestBoard board{TestSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal(board, in, uciOut, humanOut);
  terminal.run();
  EXPECT_NE(humanOut.str().find("Welcome to KittyEngineV5"), std::string::npos);
  EXPECT_NE(humanOut.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
  EXPECT_TRUE(uciOut.str().empty());
}

TEST(TerminalUI, RunStopsAtQuitOrEndOfInput) {
  std::istringstream in{"quit\n"};
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  TestBoard board{TestSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal(board, in, uciOut, humanOut);
  terminal.run();
  EXPECT_EQ(humanOut.str().find("FEN: "), humanOut.str().rfind("FEN: "));
  EXPECT_TRUE(uciOut.str().empty());
}

TEST(TerminalUI, RendersBoardAndSuppressesUnchangedFrames) {
  std::istringstream in{
    "position startpos\n"
    "position fen 7k/8/8/8/8/8/4p3/7K b - - 7 23\n"
    "quit\n"};
  std::ostringstream uciOut;
  std::ostringstream humanOut;
  TestBoard board{TestSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal(board, in, uciOut, humanOut);

  terminal.run();
  const std::string output = humanOut.str();
  const std::string startFen = "FEN: " + std::string(fen::kStartPosition) + "\n";
  const size_t firstFrameEnd = output.find(startFen);
  ASSERT_NE(firstFrameEnd, std::string::npos);
  const std::string firstFrame = output.substr(0, firstFrameEnd + startFen.size());
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

  EXPECT_EQ(output.find(startFen), output.rfind(startFen));
  const std::string secondFrame = output.substr(firstFrame.size());
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

