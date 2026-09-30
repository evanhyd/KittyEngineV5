#include "board.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_interface_policy.h"
#include "uci_protocol.h"
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  using UciBoard = Board<int, int, user_interface::TerminalInterfacePolicy>;
}

TEST(UciProtocol, DispatchesConstructorCallbacksWithoutBoard) {
  std::string calls;
  bool quit = false;
  uci::UciProtocol protocol{
    [&] { calls += 'u'; },
    [&] { calls += 'r'; },
    [&] { calls += 'n'; },
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {},
    [&] { quit = true; }
  };

  protocol.send(" unknown command\r");
  protocol.send("uci");
  protocol.send("  isready \r");
  protocol.send("ucinewgame");
  protocol.send("quit");
  EXPECT_EQ(calls, "urn");
  EXPECT_TRUE(quit);
}

TEST(UciProtocol, ParsesBothPositionFormsAndMoveLists) {
  std::vector<std::string> positions;
  uci::UciProtocol protocol{
    [] {},
    [] {},
    [] {},
    [&](std::string_view fen, std::span<const std::string_view> moves) {
      std::string received{fen};
      for (const auto move : moves) {
        received += " " + std::string(move);
      }
      positions.push_back(std::move(received));
    },
    [](std::span<const std::string_view>) {},
    [] {}
  };
  protocol.send("position startpos moves e2e4 e7e5");
  protocol.send("position fen 7k/8/8/8/8/8/8/7K b - - 4 12 moves h8g8");
  ASSERT_EQ(positions.size(), 2u);
  EXPECT_EQ(positions[0], std::string(fen::kStartPosition) + " e2e4 e7e5");
  EXPECT_EQ(positions[1], "7k/8/8/8/8/8/8/7K b - - 4 12 h8g8");
}

TEST(UciProtocol, PropagatesErrorsToItsCaller) {
  int readyCount = 0;
  uci::UciProtocol protocol{
    [] {},
    [&] { ++readyCount; },
    [] {},
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {
      throw std::logic_error("search is unfinished");
    },
    [] {}
  };

  EXPECT_THROW(protocol.send("position fen 8/8/8/8/8/8/8/8 w - -"), std::invalid_argument);
  EXPECT_THROW(protocol.send("go depth 1"), std::logic_error);
  protocol.send("isready");
  EXPECT_EQ(readyCount, 1);
}

TEST(UciIntegration, KeepsHumanViewOffProtocolOutput) {
  std::istringstream input{"uci\nisready\nquit\nuci\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{0, 0, user_interface::TerminalInterfacePolicy{input, uciOutput, humanOutput}};

  board.run();
  EXPECT_EQ(uciOutput.str(), "id name KittyEngineV5\nid author UnboxTheCat\nuciok\nreadyok\n");
  EXPECT_NE(humanOutput.str().find("Welcome to KittyEngineV5"), std::string::npos);
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
  EXPECT_EQ(humanOutput.str().find("uciok"), std::string::npos);
}

TEST(UciIntegration, ReplaysMovesAndResetsNewGame) {
  std::istringstream input{"position startpos moves e2e4 e7e5\nucinewgame\nposition startpos moves e2e4 e7e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{0, 0, user_interface::TerminalInterfacePolicy{input, uciOutput, humanOutput}};

  board.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  EXPECT_TRUE(uciOutput.str().empty());
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
}

TEST(UciIntegration, NewGameResetsBoardWithoutAnotherPositionCommand) {
  std::istringstream input{"position startpos moves e2e4\nucinewgame\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{0, 0, user_interface::TerminalInterfacePolicy{input, uciOutput, humanOutput}};

  board.run();
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_TRUE(uciOutput.str().empty());
}

TEST(UciIntegration, RollsBackInvalidPositionAndReportsGoTodo) {
  std::istringstream input{
    "position startpos moves e2e4\n"
    "position startpos moves e2e4 e7e5 e4e8\n"
    "position fen broken w - - 0 1\n"
    "go depth 2\n"
    "isready\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{0, 0, user_interface::TerminalInterfacePolicy{input, uciOutput, humanOutput}};

  board.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_NE(uciOutput.str().find("info string error: Illegal UCI move: e4e8\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("info string error: Invalid piece letter\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("info string error: go is not implemented\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("readyok\n"), std::string::npos);
}

TEST(UciIntegration, ReplaysPromotionCastlingAndEnPassant) {
  struct Case {
    const char* command;
    const char* expected;
  };
  const Case cases[] = {
    {"position fen 7k/P7/8/8/8/8/8/7K w - - 0 1 moves a7a8q\nquit\n",
     "Q6k/8/8/8/8/8/8/7K b - - 0 1"},
    {"position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1\nquit\n",
     "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1"},
    {"position fen 7k/8/8/3pP3/8/8/8/7K w - d6 0 1 moves e5d6\nquit\n",
     "7k/8/3P4/8/8/8/8/7K b - - 0 1"},
  };
  for (const Case& example : cases) {
    std::istringstream input{example.command};
    std::ostringstream uciOutput;
    std::ostringstream humanOutput;
    UciBoard board{0, 0, user_interface::TerminalInterfacePolicy{input, uciOutput, humanOutput}};
    board.run();
    EXPECT_EQ(notation::boardToFen(board.getState()), example.expected);
    EXPECT_TRUE(uciOutput.str().empty());
  }
}
