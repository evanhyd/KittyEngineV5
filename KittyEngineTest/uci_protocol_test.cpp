#include "board.h"
#include "handcraft_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_ui.h"
#include "uci_protocol.h"
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  using UciSearch = searching::NegamaxSearchPolicy<evaluation::HandCraftEvaluationPolicy>;
  using UciBoard = Board<UciSearch>;

  struct RecordingSearchPolicy {
    int* requestedDepth;
    std::vector<searching::SearchResult>* results;

    template <Color ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam param) {
      *requestedDepth = param.maxDepth;
      MoveList moves;
      state.generateMoves<ally>(moves);
      const uint64_t nodes = param.maxDepth == 3 ? 100 : param.maxDepth == 2 ? 30 : 10;
      const int32_t score = param.maxDepth == 3 ? 31'997 : param.maxDepth == 2 ? -12 : 34;
      searching::SearchResult result{
        score, moves.empty() ? std::nullopt : std::optional<Move>{moves[0]},
        nodes, std::chrono::milliseconds{10 * param.maxDepth}};
      results->push_back(result);
      return result;
    }
  };

  struct ZeroTimeSearchPolicy {
    template <Color ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam&) {
      return {0, std::nullopt, 1, std::chrono::steady_clock::duration{}};
    }
  };

  struct ScoreSequenceSearchPolicy {
    const std::vector<int32_t>* scores;

    template <Color ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam param) {
      MoveList moves;
      state.generateMoves<ally>(moves);
      return {scores->at(param.maxDepth - 1), moves.empty() ? std::nullopt : std::optional<Move>{moves[0]},
              1, std::chrono::milliseconds{1}};
    }
  };

  struct ParsedInfo {
    int depth;
    std::string scoreKind;
    int64_t score;
    int64_t timeMs;
    uint64_t nodes;
    uint64_t nps;
  };

  std::optional<ParsedInfo> parseInfoLine(const std::string& line) {
    std::istringstream stream{line};
    std::string info, depth, score, time, nodes, nps, extra;
    ParsedInfo parsed{};
    if (!(stream >> info >> depth >> parsed.depth >> score >> parsed.scoreKind >> parsed.score
                 >> time >> parsed.timeMs >> nodes >> parsed.nodes >> nps >> parsed.nps)) {
      return std::nullopt;
    }
    if (stream >> extra || info != "info" || depth != "depth" || score != "score" ||
        time != "time" || nodes != "nodes" || nps != "nps") {
      return std::nullopt;
    }
    return parsed;
  }
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

TEST(UciProtocol, DispatchesPerftAndRejectsInvalidDepth) {
  std::vector<std::pair<uint32_t, bool>> requests;
  uci::UciProtocol protocol{
    [] {}, [] {}, [] {},
    [](std::string_view, std::span<const std::string_view>) {},
    [](std::span<const std::string_view>) {},
    [] {},
    {},
    [&](uint32_t depth, bool detail) { requests.emplace_back(depth, detail); }
  };

  protocol.send("perft depth 0");
  protocol.send("perft depth 3 detail");
  EXPECT_EQ(requests, (std::vector<std::pair<uint32_t, bool>>{{0, false}, {3, true}}));
  EXPECT_THROW(protocol.send("perft"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft 3"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth -1"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth nope"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 4294967296"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 1 extra"), std::invalid_argument);
  EXPECT_THROW(protocol.send("perft depth 1 detail extra"), std::invalid_argument);
  EXPECT_EQ(requests.size(), 2u);
}

TEST(UciIntegration, KeepsHumanViewOffProtocolOutput) {
  std::istringstream input{"uci\nisready\nquit\nuci\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(uciOutput.str(), "id name KittyEngineV5\nid author UnboxTheCat\nuciok\nreadyok\n");
  EXPECT_NE(humanOutput.str().find("Welcome to KittyEngineV5"), std::string::npos);
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
  EXPECT_EQ(humanOutput.str().find("uciok"), std::string::npos);
}

TEST(UciIntegration, RunsFastAndDetailedPerftOnCurrentPosition) {
  std::istringstream input{
    "position fen " + std::string(fen::kKiwipete) + "\n"
    "perft depth 0\n"
    "perft depth 1\n"
    "perft depth 1 detail\n"
    "quit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 0, nodes 1, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 1, nodes 48, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("depth 1, nodes 48, time "), 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "    captures 8 enpassants 0 castles 2 promotions 0");
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kKiwipete);
  EXPECT_EQ(humanOutput.str().find("depth 1, nodes"), std::string::npos);
}

TEST(SearchStatistics, CountsRootAndLeafPositions) {
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 32'000}};
  MoveList legalMoves;
  board.getState().generateMoves<White>(legalMoves);

  const auto result = board.search(1, [](const searching::SearchResult&) {});
  EXPECT_EQ(result.nodesSearched, legalMoves.size() + 1);
}

TEST(UciIntegration, PassesGoDepthToSearch) {
  std::istringstream input{"go depth 3\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 3);
  ASSERT_EQ(results.size(), 3u);
  std::istringstream output{uciOutput.str()};
  uint64_t cumulativeNodes = 0;
  std::chrono::steady_clock::duration cumulativeTime{};
  std::string line;
  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    cumulativeNodes += results[i].nodesSearched;
    cumulativeTime += results[i].searchingTime;
    EXPECT_EQ(info->depth, i + 1);
    EXPECT_EQ(info->nodes, cumulativeNodes);
    EXPECT_EQ(info->timeMs, std::chrono::duration_cast<std::chrono::milliseconds>(cumulativeTime).count());
    EXPECT_EQ(info->nps, static_cast<uint64_t>(cumulativeNodes / std::chrono::duration<double>(cumulativeTime).count()));
    EXPECT_EQ(info->scoreKind, i + 1 == results.size() ? "mate" : "cp");
    if (info->scoreKind == "cp") {
      EXPECT_EQ(info->score, results[i].score);
    }
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  ASSERT_TRUE(results.back().bestMove.has_value());
  EXPECT_EQ(line, "bestmove " + notation::moveToString(*results.back().bestMove));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
}

TEST(UciIntegration, FormatsMateScoresWithinOneHundredPoints) {
  const std::vector<int32_t> scores{
    31'900, -31'900, 32'000, -32'000, 31'899, -31'899, 32'001, -32'001,
    std::numeric_limits<int32_t>::min()};
  const std::vector<int64_t> expectedScores{
    50, -50, 0, 0, 31'899, -31'899, 32'001, -32'001,
    std::numeric_limits<int32_t>::min()};
  std::istringstream input{"go depth 9\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ScoreSequenceSearchPolicy{&scores}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  std::istringstream output{uciOutput.str()};
  std::string line;
  for (size_t i = 0; i < scores.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, i + 1);
    EXPECT_EQ(info->scoreKind, i < 4 ? "mate" : "cp");
    EXPECT_EQ(info->score, expectedScores[i]);
    EXPECT_EQ(info->timeMs, 1);
    EXPECT_EQ(info->nodes, 1u);
    EXPECT_EQ(info->nps, 1000u);
  }
  MoveList moves;
  board.getState().generateMoves<White>(moves);
  ASSERT_FALSE(moves.empty());
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove " + notation::moveToString(moves[0]));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, ReplaysMovesAndResetsNewGame) {
  std::istringstream input{"position startpos moves e2e4 e7e5\nucinewgame\nposition startpos moves e2e4 e7e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  EXPECT_TRUE(uciOutput.str().empty());
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
}

TEST(UciIntegration, NewGameResetsBoardWithoutAnotherPositionCommand) {
  std::istringstream input{"position startpos moves e2e4\nucinewgame\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_TRUE(uciOutput.str().empty());
}

TEST(UciIntegration, RollsBackInvalidPositionAndSearchesToRequestedDepth) {
  std::istringstream input{
    "position startpos moves e2e4\n"
    "position startpos moves e2e4 e7e5 e4e8\n"
    "position fen broken w - - 0 1\n"
    "go depth 2\n"
    "isready\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_NE(uciOutput.str().find("info string error: Illegal UCI move: e4e8\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("info string error: Invalid piece letter\n"), std::string::npos);
  const std::string output = uciOutput.str();
  const size_t moveStart = output.find("bestmove ");
  ASSERT_NE(moveStart, std::string::npos);
  const size_t moveEnd = output.find('\n', moveStart);
  ASSERT_NE(moveEnd, std::string::npos);
  const std::string bestMove = output.substr(moveStart + 9, moveEnd - moveStart - 9);
  MoveList legalMoves;
  board.getState().generateMoves<Black>(legalMoves);
  EXPECT_TRUE(std::any_of(legalMoves.begin(), legalMoves.end(), [&](const Move& move) {
    return notation::moveToString(move) == bestMove;
  }));
  EXPECT_NE(uciOutput.str().find("readyok\n"), std::string::npos);
}

TEST(UciIntegration, RejectsInvalidDepthAndReportsNoLegalMove) {
  std::istringstream input{
    "go\n"
    "go depth\n"
    "go depth 0\n"
    "go depth nope\n"
    "position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\n"
    "go depth 2\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  const std::string output = uciOutput.str();
  std::istringstream lines{output};
  std::string line;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
    EXPECT_EQ(line.find("info string error: "), 0u);
  }
  uint64_t previousNodes = 0;
  int64_t previousTime = 0;
  for (int depth = 1; depth <= 2; ++depth) {
    ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, depth);
    EXPECT_GE(info->nodes, previousNodes);
    EXPECT_GE(info->timeMs, previousTime);
    previousNodes = info->nodes;
    previousTime = info->timeMs;
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(lines, line)));
  EXPECT_EQ(line, "bestmove 0000");
  EXPECT_FALSE(static_cast<bool>(std::getline(lines, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
}

TEST(UciIntegration, ReportsCheckmateWithoutLegalMove) {
  std::istringstream input{"position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1\ngo depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  EXPECT_EQ(info->scoreKind, "mate");
  EXPECT_EQ(info->score, 0);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
}

TEST(UciIntegration, ReportsZeroNpsWhenElapsedTimeIsZero) {
  std::istringstream input{"go depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ZeroTimeSearchPolicy{}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->timeMs, 0);
  EXPECT_EQ(info->nps, 0u);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
}

TEST(UciIntegration, PlayAcceptsLegalMoveAndRejectsIllegalMove) {
  std::istringstream input{"play e2e4\nplay e2e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
  EXPECT_EQ(uciOutput.str(), "info string error: Illegal UCI move: e2e5\n");
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
    UciBoard board{UciSearch{evaluation::HandCraftEvaluationPolicy{}, 50}};
    user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};
    terminal.run();
    EXPECT_EQ(notation::boardToFen(board.getState()), example.expected);
    EXPECT_TRUE(uciOutput.str().empty());
  }
}
