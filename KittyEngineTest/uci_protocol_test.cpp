#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "notation.h"
#include "position_fens.h"
#include "terminal_ui.h"
#include "test_move_callback.h"
#include "test_search_config.h"
#include "uci_protocol.h"
#include <array>
#include <deque>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  using UciSearch = searching::NegamaxSearchPolicy<evaluation::HandCraftEvaluationPolicy>;
  using TestTimePolicy = time_control::EqualPercentageTimeControlPolicy;
  using UciBoard = Board<UciSearch, TestTimePolicy>;
  constexpr int32_t kMateMagnitude = evaluation::kCheckmateWinScore;
  constexpr int32_t kMateWindow = 100;
  constexpr size_t kTestTranspositionEntries = 1024;
  constexpr float kTestTimePercentage = 0.05f;

  auto makeUciSearch(int32_t aspirationWindow, size_t ttEntries = kTestTranspositionEntries) {
    return UciSearch{evaluation::HandCraftEvaluationPolicy{}, testing_support::makeTestSearchConfig(), aspirationWindow, ttEntries};
  }

  struct RecordingSearchPolicy {
    int* requestedDepth;
    std::vector<searching::SearchResult>* results;
    std::vector<std::vector<Move>>* pvSnapshots = nullptr;
    int* resetCalls = nullptr;
    std::vector<size_t>* historySizes = nullptr;
    std::deque<searching::PVLine> pvLinesStorage;

    void reset() {
      if (resetCalls) {
        ++*resetCalls;
      }
    }

    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      *requestedDepth = param.maxDepth;
      if (historySizes) {
        historySizes->push_back(param.history.size());
      }
      MoveList moves;
      state.generateMoves<ally>(moves);
      searching::PVLine pvLine;
      if (!moves.empty()) {
        pvLine.push(moves[0]);
        if (param.maxDepth > 1) {
          BoardState nextState = state;
          nextState.makeMove<ally>(moves[0], testing_support::NoOpMoveCallback{});
          MoveList replies;
          nextState.generateMoves<getOtherSide(ally)>(replies);
          if (!replies.empty()) {
            pvLine.push(replies[0]);
          }
        }
      }
      auto& stored = pvLinesStorage.emplace_back(std::move(pvLine));
      const uint64_t nodes = param.maxDepth == 3 ? 100 : param.maxDepth == 2 ? 30 : 10;
      const int32_t score = param.maxDepth == 3 ? kMateMagnitude - 3 : param.maxDepth == 2 ? -12 : 34;
      searching::SearchResult result{
        .score = score,
        .move = moves.empty() ? Move{} : moves[0],
        .pvLine = std::span<const Move>{stored.data(), stored.size()},
        .nodes = nodes,
        .searchingTime = std::chrono::milliseconds{10 * param.maxDepth}};
      if (pvSnapshots) {
        pvSnapshots->emplace_back(stored.begin(), stored.end());
      }
      results->push_back(result);
      return result;
    }
  };

  struct ZeroTimeSearchPolicy {
    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam&) {
      return {.score = 0, .move = Move{}, .pvLine = {}, .nodes = 1, .searchingTime = std::chrono::nanoseconds{}};
    }
  };

  struct SlowSearchPolicy {
    int* requestedDepth;
    std::chrono::milliseconds pause;

    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      *requestedDepth = param.maxDepth;
      std::this_thread::sleep_for(pause);
      MoveList moves;
      state.generateMoves<ally>(moves);
      return {.score = 0, .move = moves.empty() ? Move{} : moves[0], .pvLine = {}, .nodes = 1, .searchingTime = pause};
    }
  };

  struct SubMillisecondSearchPolicy {
    static constexpr uint64_t kNodes = 100;
    static constexpr auto kDuration = std::chrono::microseconds{250};

    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam&) {
      return {.score = 0, .move = Move{}, .pvLine = {}, .nodes = kNodes, .searchingTime = kDuration};
    }
  };

  struct ScoreSequenceSearchPolicy {
    static constexpr uint64_t kNodesPerDepth = 1;
    static constexpr auto kSearchDuration = std::chrono::milliseconds{1};
    const std::vector<int32_t>* scores;

    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam param) {
      MoveList moves;
      state.generateMoves<ally>(moves);
      return {.score = scores->at(param.maxDepth - 1),
              .move = moves.empty() ? Move{} : moves[0],
              .pvLine = {},
              .nodes = kNodesPerDepth,
              .searchingTime = kSearchDuration};
    }
  };

  struct DummyPVSearchPolicy {
    searching::PVLine pvToReturn;

    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t score) const noexcept {
      const int32_t absScore = std::abs(score);
      return absScore >= evaluation::kCheckmateWinScore - searching::kMaxDepthHardCutoff && absScore <= evaluation::kCheckmateWinScore;
    }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState&, const searching::SearchParam&) {
      return {.score = 0,
              .move = pvToReturn.empty() ? Move{} : pvToReturn[0],
              .pvLine = std::span<const Move>{pvToReturn.data(), pvToReturn.size()},
              .nodes = 1,
              .searchingTime = std::chrono::nanoseconds{}};
    }
  };

  struct ParsedInfo {
    int depth = 0;
    int reachedPly = 0;
    std::string scoreKind;
    int64_t score = 0;
    int64_t timeMs = 0;
    uint64_t nodes = 0;
    uint64_t nps = 0;
    std::vector<std::string> pv;
  };

  std::optional<ParsedInfo> parseInfoLine(const std::string& line) {
    std::istringstream stream{line};
    std::string token;
    ParsedInfo parsed{};
    if (!(stream >> token) || token != "info") {
      return std::nullopt;
    }
    bool hasDepth = false;
    bool hasScore = false;
    bool hasTime = false;
    bool hasNodes = false;
    bool hasNps = false;

    while (stream >> token) {
      if (token == "depth") {
        if (!(stream >> parsed.depth)) return std::nullopt;
        hasDepth = true;
      } else if (token == "seldepth") {
        if (!(stream >> parsed.reachedPly)) return std::nullopt;
      } else if (token == "score") {
        if (!(stream >> parsed.scoreKind >> parsed.score)) return std::nullopt;
        hasScore = true;
      } else if (token == "time") {
        if (!(stream >> parsed.timeMs)) return std::nullopt;
        hasTime = true;
      } else if (token == "nodes") {
        if (!(stream >> parsed.nodes)) return std::nullopt;
        hasNodes = true;
      } else if (token == "nps") {
        if (!(stream >> parsed.nps)) return std::nullopt;
        hasNps = true;
      } else if (token == "hashfull") {
        uint64_t hf = 0;
        if (!(stream >> hf)) return std::nullopt;
      } else if (token == "pv") {
        while (stream >> token) {
          if (token == "string") break;
          parsed.pv.push_back(token);
        }
        if (parsed.pv.empty()) return std::nullopt;
        break;
      }
    }

    if (!hasDepth || !hasScore || !hasTime || !hasNodes || !hasNps) {
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
  protocol.send("position kiwipete moves a6b7");
  ASSERT_EQ(positions.size(), 3u);
  EXPECT_EQ(positions[0], std::string(fen::kStartPosition) + " e2e4 e7e5");
  EXPECT_EQ(positions[1], "7k/8/8/8/8/8/8/7K b - - 4 12 h8g8");
  EXPECT_EQ(positions[2], std::string(fen::kKiwipete) + " a6b7");
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
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
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
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
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
  UciBoard board{makeUciSearch(32'000), TestTimePolicy{kTestTimePercentage}};
  MoveList legalMoves;
  board.getState().generateMoves<White>(legalMoves);

  searching::SearchResult result{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { result = r; });
  EXPECT_GE(result.nodes, 1u);
  EXPECT_LE(result.nodes, legalMoves.size() + 1);
  ASSERT_TRUE(static_cast<bool>(result.move));
}

TEST(SearchStatistics, ReusesRootTranspositionWithNanosecondDuration) {
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
  searching::SearchResult first{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { first = r; });
  searching::SearchResult cached{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { cached = r; });

  ASSERT_TRUE(static_cast<bool>(first.move));
  ASSERT_TRUE(static_cast<bool>(cached.move));
  EXPECT_EQ(cached.score, first.score);
  EXPECT_EQ(notation::moveToString(cached.move), notation::moveToString(first.move));
  EXPECT_LE(cached.nodes, first.nodes);
  EXPECT_GE(cached.searchingTime.count(), 0);
}

TEST(SearchFrame, ReducedAndNullMoveSearchesKeepLegalPVAndWarmTTResult) {
  UciBoard board{makeUciSearch(80, 1u << 16),
                 TestTimePolicy{kTestTimePercentage}};
  searching::SearchResult cold{};
  board.search(5, std::nullopt, [&](const searching::SearchResult& r) { cold = r; });
  ASSERT_TRUE(static_cast<bool>(cold.move));
  ASSERT_FALSE(cold.pvLine.empty());
  EXPECT_EQ(cold.pvLine.front(), cold.move);
  EXPECT_LE(cold.pvLine.size(), 5u);

  BoardState pvState = board.getState();
  for (const Move move : cold.pvLine) {
    MoveList legalMoves;
    if (pvState.getSide() == White) {
      pvState.generateMoves<White>(legalMoves);
      ASSERT_NE(std::find(legalMoves.begin(), legalMoves.end(), move), legalMoves.end());
      pvState.makeMove<White>(move, testing_support::NoOpMoveCallback{});
    } else {
      pvState.generateMoves<Black>(legalMoves);
      ASSERT_NE(std::find(legalMoves.begin(), legalMoves.end(), move), legalMoves.end());
      pvState.makeMove<Black>(move, testing_support::NoOpMoveCallback{});
    }
  }

  const std::vector<Move> coldPV(cold.pvLine.begin(), cold.pvLine.end());
  searching::SearchResult warm{};
  board.search(5, std::nullopt, [&](const searching::SearchResult& r) { warm = r; });
  ASSERT_TRUE(static_cast<bool>(warm.move));
  EXPECT_EQ(warm.score, cold.score);
  EXPECT_EQ(warm.move, cold.move);
  EXPECT_EQ(warm.pvLine.size(), coldPV.size());
  EXPECT_TRUE(std::equal(warm.pvLine.begin(), warm.pvLine.end(), coldPV.begin()));
}

TEST(SearchFrame, NullMovePruningRemainingDepthAndCutoffInvariants) {
  UciBoard board{makeUciSearch(80, 1u << 16),
                 TestTimePolicy{kTestTimePercentage}};
  // Position with comfortable advantage where NMP is triggered
  board.setPosition("r1bq1rk1/ppp2ppp/2n5/3np3/8/2NBPN2/PPP2PPP/R2QK2R w KQ - 0 1");
  searching::SearchResult res{};
  board.search(6, std::nullopt, [&](const searching::SearchResult& r) { res = r; });
  EXPECT_TRUE(static_cast<bool>(res.move));
  EXPECT_FALSE(res.pvLine.empty());
  EXPECT_EQ(res.move, res.pvLine.front());
}

TEST(SearchFrame, MateDistanceUsesActualPlyAcrossDepths) {
  BoardState state{"7k/8/5KQ1/8/8/8/8/8 w - - 0 1"};
  UciSearch search{makeUciSearch(80, 1u << 16)};
  searching::PositionHistory history;
  history.push(state.getHash());

  for (const int depth : {1, 4}) {
    const searching::SearchParam param{.maxDepth = depth, .history = history, .isTimeUpHard = nullptr};
    const auto result = search.search<White>(state, param);
    ASSERT_TRUE(static_cast<bool>(result.move));
    EXPECT_EQ(result.score, evaluation::kCheckmateWinScore - 1);
    ASSERT_FALSE(result.pvLine.empty());
    EXPECT_EQ(result.pvLine.front(), result.move);
  }
}

TEST(BoardSearch, StopsAfterWinningMate) {
  const std::vector<int32_t> scores{kMateMagnitude - 1, 0, 0};
  Board board{ScoreSequenceSearchPolicy{&scores}, TestTimePolicy{kTestTimePercentage}};
  int completedDepths = 0;
  searching::SearchResult result{};
  board.search(3, std::nullopt, [&](const searching::SearchResult& r) {
    result = r;
    ++completedDepths;
  });
  EXPECT_EQ(completedDepths, 1);
  EXPECT_EQ(result.score, scores.front());
}

TEST(BoardSearch, ContinuesForLosingMateAndScoresOutsideMateRange) {
  for (const std::vector<int32_t> scores : {
         std::vector<int32_t>{-kMateMagnitude + 1, -kMateMagnitude, 0},
         std::vector<int32_t>{kMateMagnitude - searching::kMaxDepthHardCutoff - 1,
                              kMateMagnitude + 1, 0}}) {
    Board board{ScoreSequenceSearchPolicy{&scores}, TestTimePolicy{kTestTimePercentage}};
    int completedDepths = 0;
    searching::SearchResult result{};
    board.search(3, std::nullopt, [&](const searching::SearchResult& r) {
      result = r;
      ++completedDepths;
    });
    EXPECT_EQ(completedDepths, 3);
    EXPECT_EQ(result.score, 0);
  }
}

TEST(UciIntegration, StopsAfterMateInOneAndReportsBestMove) {
  std::istringstream input{"position fen 7k/8/5KQ1/8/8/8/8/8 w - - 0 1\ngo depth 4\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{makeUciSearch(80, 1u << 16),
                 TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  EXPECT_EQ(info->scoreKind, "mate");
  EXPECT_EQ(info->score, 1);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("bestmove "), 0u);
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, PassesGoDepthToSearch) {
  std::istringstream input{"go depth 3\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  std::vector<std::vector<Move>> pvSnapshots;
  Board board{RecordingSearchPolicy{&requestedDepth, &results, &pvSnapshots}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 3);
  ASSERT_EQ(results.size(), 3u);
  std::istringstream output{uciOutput.str()};
  std::string line;
  uint64_t totalNodes = 0;
  int64_t previousTimeMs = 0;
  for (size_t i = 0; i < results.size(); ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, i + 1);
    totalNodes += results[i].nodes;
    EXPECT_EQ(info->nodes, totalNodes);
    EXPECT_GE(info->timeMs, previousTimeMs);
    if (info->timeMs > 0 && info->timeMs < 1000) {
      EXPECT_GT(info->nps, info->nodes);
    }
    previousTimeMs = info->timeMs;
    EXPECT_EQ(info->scoreKind, i + 1 == results.size() ? "mate" : "cp");
    if (info->scoreKind == "cp") {
      EXPECT_EQ(info->score, results[i].score);
    }
    const auto& expectedPV = pvSnapshots[i];
    ASSERT_EQ(info->pv.size(), expectedPV.size());
    for (size_t moveIndex = 0; moveIndex < expectedPV.size(); ++moveIndex) {
      EXPECT_EQ(info->pv[moveIndex], notation::moveToString(expectedPV[moveIndex]));
    }
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  ASSERT_TRUE(static_cast<bool>(results.back().move));
  EXPECT_EQ(line, "bestmove " + notation::moveToString(results.back().move));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
}

TEST(UciIntegration, StopsTimedSearchAfterFirstCompletedDepthWhenBudgetIsZero) {
  std::istringstream input{"go wtime 0 btime 10000 winc 0 binc 1000 movestogo 9999\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  ASSERT_EQ(results.size(), 1u);
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  ASSERT_TRUE(static_cast<bool>(results.back().move));
  EXPECT_EQ(line, "bestmove " + notation::moveToString(results.back().move));
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
}

TEST(UciIntegration, UsesMoveTimeWithoutClocks) {
  std::istringstream input{"go movetime 100\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  Board board{SlowSearchPolicy{&requestedDepth, std::chrono::milliseconds{75}}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  EXPECT_EQ(uciOutput.str().find("info string error:"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("bestmove "), std::string::npos);
}

TEST(UciIntegration, MoveTimeOverridesClockAndAllowsDepthLimit) {
  std::istringstream input{"go depth 2 movetime 0 wtime 100000 btime 100000\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  EXPECT_EQ(results.size(), 1u);
  EXPECT_EQ(uciOutput.str().find("info string error:"), std::string::npos);
}

TEST(UciIntegration, CapsRequestedSearchDepth) {
  const int depthBeyondLimit = searching::kMaxDepthHardCutoff + 1;
  const std::vector<int32_t> scores(searching::kMaxDepthHardCutoff, 0);
  std::istringstream input{"go depth " + std::to_string(depthBeyondLimit) + "\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ScoreSequenceSearchPolicy{&scores}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  std::istringstream output{uciOutput.str()};
  std::string line;
  for (int depth = 1; depth <= searching::kMaxDepthHardCutoff; ++depth) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, depth);
  }
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("bestmove "), 0u);
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(BoardSearch, SearchReturnsPVLineFromPolicy) {
  DummyPVSearchPolicy policy;
  policy.pvToReturn.push(Move{E2, E4, Pawn, NoPiece, 0});
  policy.pvToReturn.push(Move{E7, E5, Pawn, NoPiece, 0});

  Board board{policy, TestTimePolicy{kTestTimePercentage}};
  searching::SearchResult result{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { result = r; });
  ASSERT_EQ(result.pvLine.size(), 2u);
  EXPECT_EQ(result.pvLine[0], policy.pvToReturn[0]);
  EXPECT_EQ(result.pvLine[1], policy.pvToReturn[1]);
}

TEST(UciIntegration, SelectsBlackIncrementForTimedSearch) {
  std::istringstream input{
    "position fen 7k/8/8/8/8/8/8/7K b - - 0 1\n"
    "go depth 2 wtime 0 btime 0 winc 0 binc 1000\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 2);
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(uciOutput.str().find("info string error:"), std::string::npos);
}

TEST(UciIntegration, UsesLastDepthDurationToAvoidStartingAnotherDepth) {
  std::istringstream input{"go wtime 2000 btime 2000 winc 0 binc 0\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  Board board{SlowSearchPolicy{&requestedDepth, std::chrono::milliseconds{75}}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 1);
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line.find("bestmove "), 0u);
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, RejectsIncompleteOrInvalidTimeControls) {
  std::istringstream input{
    "go wtime 100\n"
    "go depth 2 winc 10\n"
    "go wtime -1 btime 100\n"
    "go wtime nope btime 100\n"
    "go wtime 100 btime 100 btime 200\n"
    "go wtime 100 btime 100 winc\n"
    "go wtime 100 btime 100 movestogo nope\n"
    "go wtime 100 btime 100 movestogo 1 movestogo 2\n"
    "go movetime\n"
    "go movetime -1\n"
    "go movetime nope\n"
    "go movetime 100 movetime 200\n"
    "quit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(requestedDepth, 0);
  EXPECT_TRUE(results.empty());
  std::istringstream output{uciOutput.str()};
  std::string line;
  for (int i = 0; i < 12; ++i) {
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    EXPECT_EQ(line.find("info string error:"), 0u);
  }
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, FormatsMateScoresWithinOneHundredPoints) {
  const std::vector<int32_t> scores{
    kMateMagnitude - kMateWindow, -(kMateMagnitude - kMateWindow),
    kMateMagnitude, -kMateMagnitude,
    kMateMagnitude - kMateWindow - 1, -(kMateMagnitude - kMateWindow - 1),
    kMateMagnitude + 1, -(kMateMagnitude + 1),
    std::numeric_limits<int32_t>::min()};
  const std::vector<int64_t> expectedScores{
    (kMateWindow + 1) / 2, -(kMateWindow + 1) / 2, 0, 0,
    kMateMagnitude - kMateWindow - 1, -(kMateMagnitude - kMateWindow - 1),
    kMateMagnitude + 1, -(kMateMagnitude + 1),
    std::numeric_limits<int32_t>::min()};
  for (size_t i = 0; i < scores.size(); ++i) {
    SCOPED_TRACE(i);
    const std::vector<int32_t> singleScore{scores[i]};
    std::istringstream input{"go depth 1\nquit\n"};
    std::ostringstream uciOutput;
    std::ostringstream humanOutput;
    Board board{ScoreSequenceSearchPolicy{&singleScore}, TestTimePolicy{kTestTimePercentage}};
    user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};
    terminal.run();

    std::istringstream output{uciOutput.str()};
    std::string line;
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    const auto info = parseInfoLine(line);
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->depth, 1);
    EXPECT_EQ(info->scoreKind, i < 4 ? "mate" : "cp");
    EXPECT_EQ(info->score, expectedScores[i]);
    EXPECT_TRUE(info->pv.empty());
    EXPECT_GE(info->timeMs, 0);
    EXPECT_EQ(info->nodes, ScoreSequenceSearchPolicy::kNodesPerDepth);
    MoveList moves;
    board.getState().generateMoves<White>(moves);
    ASSERT_FALSE(moves.empty());
    ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
    EXPECT_EQ(line, "bestmove " + notation::moveToString(moves[0]));
    EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
  }
}

TEST(UciIntegration, ReplaysMovesAndResetsNewGame) {
  std::istringstream input{"position startpos moves e2e4 e7e5\nucinewgame\nposition startpos moves e2e4 e7e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  EXPECT_TRUE(uciOutput.str().empty());
  EXPECT_NE(humanOutput.str().find("FEN: " + std::string(fen::kStartPosition)), std::string::npos);
}

TEST(UciIntegration, NewGameResetsBoardToStartPosition) {
  std::istringstream input{"position startpos moves e2e4\nucinewgame\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_TRUE(uciOutput.str().empty());
}

TEST(UciIntegration, NewGameResetsSearchPolicyAndBoardHistory) {
  std::istringstream input{
    "position startpos moves e2e4\nucinewgame\ngo depth 1\nposition startpos\ngo depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  int requestedDepth = 0;
  int resetCalls = 0;
  std::vector<size_t> historySizes;
  std::vector<searching::SearchResult> results;
  Board board{RecordingSearchPolicy{&requestedDepth, &results, nullptr, &resetCalls, &historySizes},
              TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();

  EXPECT_EQ(resetCalls, 1);
  EXPECT_EQ(historySizes, (std::vector<size_t>{1, 1}));
  EXPECT_EQ(requestedDepth, 1);
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_NE(uciOutput.str().find("bestmove "), std::string::npos);
}

TEST(BoardReset, ClearsSearchHistoryAndPolicyButKeepsPosition) {
  int requestedDepth = 0;
  int resetCalls = 0;
  std::vector<searching::SearchResult> results;
  std::vector<size_t> historySizes;
  Board board{RecordingSearchPolicy{
    &requestedDepth, &results, nullptr, &resetCalls, &historySizes},
    TestTimePolicy{kTestTimePercentage}};

  board.playMove("e2e4");
  searching::SearchResult first{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { first = r; });
  ASSERT_FALSE(first.pvLine.empty());
  const std::string position = notation::boardToFen(board.getState());

  board.newGame();
  board.search(1, std::nullopt, [](const searching::SearchResult&) {});

  EXPECT_EQ(resetCalls, 1);
  EXPECT_EQ(notation::boardToFen(board.getState()), fen::kStartPosition);
  EXPECT_EQ(historySizes, (std::vector<size_t>{2, 1}));
}

TEST(UciIntegration, ReportsErrorOnInvalidPositionAndSearchesToRequestedDepth) {
  std::istringstream input{
    "position startpos moves e2e4\n"
    "position startpos moves e2e4 e7e5 e4e8\n"
    "position fen broken w - - 0 1\n"
    "go depth 2\n"
    "isready\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  EXPECT_EQ(notation::boardToFen(board.getState()),
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
  EXPECT_NE(uciOutput.str().find("info string error: Illegal UCI move: e4e8\n"), std::string::npos);
  EXPECT_NE(uciOutput.str().find("info string error: Invalid piece letter\n"), std::string::npos);
  const std::string output = uciOutput.str();
  const size_t moveStart = output.find("bestmove ");
  ASSERT_NE(moveStart, std::string::npos);
  const size_t moveEnd = output.find('\n', moveStart);
  ASSERT_NE(moveEnd, std::string::npos);
  const std::string bestMove = output.substr(moveStart + 9, moveEnd - moveStart - 9);
  MoveList legalMoves;
  board.getState().generateMoves<White>(legalMoves);
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
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
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
    EXPECT_TRUE(info->pv.empty());
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
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
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

TEST(UciIntegration, ReportsNpsForFastSearch) {
  std::istringstream input{"go depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{ZeroTimeSearchPolicy{}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_GE(info->timeMs, 0);
  EXPECT_GT(info->nps, info->nodes);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
}

TEST(UciIntegration, ReportsElapsedTimeAndNps) {
  std::istringstream input{"go depth 1\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  Board board{SubMillisecondSearchPolicy{}, TestTimePolicy{kTestTimePercentage}};
  user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};

  terminal.run();
  std::istringstream output{uciOutput.str()};
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  const auto info = parseInfoLine(line);
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->depth, 1);
  EXPECT_GE(info->timeMs, 0);
  EXPECT_EQ(info->nodes, SubMillisecondSearchPolicy::kNodes);
  EXPECT_GT(info->nps, info->nodes);
  ASSERT_TRUE(static_cast<bool>(std::getline(output, line)));
  EXPECT_EQ(line, "bestmove 0000");
  EXPECT_FALSE(static_cast<bool>(std::getline(output, line)));
}

TEST(UciIntegration, PlayAcceptsLegalMoveAndRejectsIllegalMove) {
  std::istringstream input{"play e2e4\nplay e2e5\nquit\n"};
  std::ostringstream uciOutput;
  std::ostringstream humanOutput;
  UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
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
    UciBoard board{makeUciSearch(50), TestTimePolicy{kTestTimePercentage}};
    user_interface::TerminalUI terminal{board, input, uciOutput, humanOutput};
    terminal.run();
    EXPECT_EQ(notation::boardToFen(board.getState()), example.expected);
    EXPECT_TRUE(uciOutput.str().empty());
  }
}


TEST(SearchConfigYaml, SerializesDefaultConfigToStream) {
  searching::NegamaxSearchConfig config = testing_support::makeTestSearchConfig();
  std::ostringstream os;
  config.serialize(os);
  const std::string yaml = os.str();

  EXPECT_FALSE(yaml.empty());
  EXPECT_NE(yaml.find("aspiration_window: " + std::to_string(config.aspirationWindow)), std::string::npos);
  EXPECT_NE(yaml.find("improving_threshold: " + std::to_string(config.improvingThreshold)), std::string::npos);
  EXPECT_NE(yaml.find("reverse_futility_pruning:"), std::string::npos);
  EXPECT_NE(yaml.find("null_move_pruning:"), std::string::npos);
  EXPECT_NE(yaml.find("late_move_reductions:"), std::string::npos);
  EXPECT_NE(yaml.find("singular_extension:"), std::string::npos);
}

TEST(SearchConfigYaml, DeserializesFromStreamAndRoundTrips) {
  searching::NegamaxSearchConfig config{};

  constexpr std::string_view customYaml = R"(
aspiration_window: 85
improving_threshold: 35

reverse_futility_pruning:
  depth: 5
  multiplier: 120
  base: 10
  improving_bonus: 45

null_move_pruning:
  depth: 6
  base: 4
  depth_divisor: 5
  eval_divisor: 250
  max_eval_bonus: 3
)";

  std::istringstream is{std::string(customYaml)};
  config.deserialize(is);

  EXPECT_EQ(config.aspirationWindow, 85);
  EXPECT_EQ(config.improvingThreshold, 35);
  EXPECT_EQ(config.reverseFutilityDepth, 5);
  EXPECT_EQ(config.reverseFutilityMultiplier, 120);
  EXPECT_EQ(config.reverseFutilityBase, 10);
  EXPECT_EQ(config.reverseFutilityImprovingBonus, 45);
  EXPECT_EQ(config.nullMoveDepth, 6);
  EXPECT_EQ(config.nullMoveBase, 4);
  EXPECT_EQ(config.nullMoveDepthDivisor, 5);
  EXPECT_EQ(config.nullMoveEvalDivisor, 250);
  EXPECT_EQ(config.nullMoveMaxEvalBonus, 3);

  std::ostringstream os1;
  config.serialize(os1);
  const std::string roundTrip1 = os1.str();

  searching::NegamaxSearchConfig config2{};
  std::istringstream is2{roundTrip1};
  config2.deserialize(is2);

  std::ostringstream os2;
  config2.serialize(os2);
  const std::string roundTrip2 = os2.str();

  EXPECT_EQ(roundTrip1, roundTrip2);
}

TEST(InSearchTimeCheck, SearchAbortsMidDepthWhenTimeExpires) {
  BoardState state{fen::kStartPosition};
  UciSearch search{makeUciSearch(80, 1u << 16)};
  searching::PositionHistory history;
  history.push(state.getHash());

  int checkCount = 0;
  const searching::SearchParam param{
    .maxDepth = 20,
    .history = history,
    .isTimeUpHard = [&checkCount]() {
      ++checkCount;
      return true;
    }
  };

  const auto result = search.search<White>(state, param);
  EXPECT_TRUE(result.aborted);
  EXPECT_GT(checkCount, 0);
}

TEST(TimeControlPolicyTest, SoftAndHardTimeLimits) {
  TestTimePolicy policy{0.05f};
  time_control::TimeControl tc;
  tc.moveTime = std::chrono::milliseconds{40};
  policy.set(tc);

  EXPECT_FALSE(policy.isTimeUpHard(White));

  std::this_thread::sleep_for(std::chrono::milliseconds{45});
  EXPECT_TRUE(policy.isTimeUpHard(White));
}

namespace {
  struct TimeParamRecordingPolicy {
    std::vector<bool>* hasTimeCheckPerDepth = nullptr;

    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t) const noexcept { return false; }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      if (hasTimeCheckPerDepth) {
        hasTimeCheckPerDepth->push_back(static_cast<bool>(param.isTimeUpHard));
      }
      MoveList moves;
      state.generateMoves<ally>(moves);
      return searching::SearchResult{
        .score = 0,
        .move = moves.empty() ? Move{} : moves[0],
        .pvLine = {},
        .nodes = 1,
        .searchingTime = std::chrono::nanoseconds{1},
        .probes = 0,
        .hits = 0,
        .plyReached = param.maxDepth,
        .aborted = false
      };
    }
  };
}

TEST(InSearchTimeCheck, Depth10PreconditionEnforced) {
  std::vector<bool> hasTimeCheckPerDepth;
  TimeParamRecordingPolicy policy{&hasTimeCheckPerDepth};
  TestTimePolicy timePolicy{kTestTimePercentage};
  Board board{policy, timePolicy};

  board.search(12, std::nullopt, [](const searching::SearchResult&) {});

  ASSERT_EQ(hasTimeCheckPerDepth.size(), 12u);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_FALSE(hasTimeCheckPerDepth[i]) << "Depth " << (i + 1) << " should not have hard time check";
  }
  EXPECT_TRUE(hasTimeCheckPerDepth[10]) << "Depth 11 should have hard time check";
  EXPECT_TRUE(hasTimeCheckPerDepth[11]) << "Depth 12 should have hard time check";
}

namespace {
  struct AbortAtDepth2Policy {
    void reset() {}
    void resetTurn() {}
    bool isCheckmateScore(int32_t) const noexcept { return false; }
    template <bool Add>
    void markPiece(Side, Piece, Square) {}

    template <Side ally>
    searching::SearchResult search(BoardState& state, const searching::SearchParam& param) {
      MoveList moves;
      state.generateMoves<ally>(moves);
      if (param.maxDepth == 1) {
        return searching::SearchResult{
          .score = 50,
          .move = moves.empty() ? Move{} : moves[0],
          .pvLine = {},
          .nodes = 20,
          .searchingTime = std::chrono::milliseconds{1},
          .probes = 0,
          .hits = 0,
          .plyReached = 1,
          .aborted = false
        };
      }
      return searching::SearchResult{
        .score = 0,
        .move = Move{},
        .pvLine = {},
        .nodes = 100,
        .searchingTime = std::chrono::milliseconds{10},
        .probes = 0,
        .hits = 0,
        .plyReached = 1,
        .aborted = true
      };
    }
  };
}

TEST(InSearchTimeCheck, BoardIterativeDeepeningDiscardsAbortedDepth) {
  Board board{AbortAtDepth2Policy{}, TestTimePolicy{kTestTimePercentage}};
  int callbackCount = 0;
  searching::SearchResult lastResult{};
  board.search(5, std::nullopt, [&](const searching::SearchResult& r) {
    ++callbackCount;
    lastResult = r;
  });

  EXPECT_EQ(callbackCount, 1);
  EXPECT_EQ(lastResult.score, 50);
  EXPECT_FALSE(lastResult.aborted);
}

TEST(InSearchTimeCheck, AbortReturnPathPreservesBoardAndHistoryState) {
  BoardState state{fen::kKiwipete};
  const uint64_t expectedHash = state.getHash();
  const Side expectedSide = state.getSide();
  std::array<std::array<Bitboard, kPieceSize>, kSideSize> expectedBitboards{};
  for (Side s : {White, Black}) {
    for (Piece p : {Pawn, Knight, Bishop, Rook, Queen, King}) {
      expectedBitboards[s][p] = state.getPieces(s, p);
    }
  }

  UciSearch search{makeUciSearch(80, 1u << 16)};
  searching::PositionHistory history;
  history.push(expectedHash);
  const size_t expectedHistorySize = history.size();

  // Abort mid-search after exploring many nodes deep in the tactical tree
  int nodeCounter = 0;
  const searching::SearchParam param{
    .maxDepth = 20,
    .history = history,
    .isTimeUpHard = [&nodeCounter]() {
      ++nodeCounter;
      return nodeCounter >= 500;
    }
  };

  const auto result = search.search<White>(state, param);
  EXPECT_TRUE(result.aborted);
  EXPECT_GE(nodeCounter, 500);

  // 1. Verify hash, side, and history are completely unchanged
  EXPECT_EQ(state.getHash(), expectedHash);
  EXPECT_EQ(state.getSide(), expectedSide);
  EXPECT_EQ(history.size(), expectedHistorySize);
  EXPECT_EQ(history.front(), expectedHash);

  // 2. Verify all piece bitboards are completely intact
  for (Side s : {White, Black}) {
    for (Piece p : {Pawn, Knight, Bishop, Rook, Queen, King}) {
      EXPECT_EQ(state.getPieces(s, p), expectedBitboards[s][p]);
    }
  }

  // 3. Verify that a subsequent normal search runs correctly and cleanly on this state
  const searching::SearchParam normalParam{
    .maxDepth = 3,
    .history = history,
    .isTimeUpHard = nullptr
  };
  const auto normalResult = search.search<White>(state, normalParam);
  EXPECT_FALSE(normalResult.aborted);
  EXPECT_TRUE(static_cast<bool>(normalResult.move));
  EXPECT_GT(normalResult.nodes, 0u);
}
