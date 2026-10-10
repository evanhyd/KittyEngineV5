#include "board.h"
#include "tablebase.h"
#include "tablebase_uci.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "terminal_ui.h"
#include "test_move_callback.h"
#include "syzygy_test_path.h"
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <gtest/gtest.h>

using namespace bb;
using testing_support::syzygyTestPath;

namespace {
  MoveList legalMoves(const BoardState& state) {
    MoveList moves;
    if (state.getSideToMove() == White) {
      state.generateMoves<White>(moves);
    } else {
      state.generateMoves<Black>(moves);
    }
    return moves;
  }
  using Search = searching::NegamaxSearchPolicy<evaluation::HandCraftEvaluationPolicy>;
  using TestBoard = Board<Search, time_control::EqualPercentageTimeControlPolicy>;
  class SyzygyFiles : public ::testing::Test {
  protected:
    tablebase::Service service;
    void SetUp() override {
      const auto path = syzygyTestPath();
      if (path.empty()) {
        GTEST_SKIP() << "Set KITTY_SYZYGY_TEST_PATH to verified 3-4 piece WDL+DTZ fixtures";
      }
      service.setOption("SyzygyPath", path);
      ASSERT_GE(service.largest(), 4u);
    }
    TestBoard board() {
      return TestBoard{Search{evaluation::HandCraftEvaluationPolicy{}, 80, 1u << 14, &service},
        time_control::EqualPercentageTimeControlPolicy{0.05f}};
    }
  };
}

TEST(Syzygy, DisabledAndUnavailableAreNotDraws) {
  tablebase::Service service;
  EXPECT_FALSE(service.enabled());
  BoardState state("7k/8/5K2/8/8/8/8/R7 w - - 0 1");
  EXPECT_FALSE(service.probeWdl(state));
  service.setOption("SyzygyPath", "");
  EXPECT_EQ(service.largest(), 0u);
  EXPECT_THROW(service.setOption("SyzygyProbeLimit", "8"), std::invalid_argument);
  EXPECT_THROW(service.setOption("SyzygyProbeDepth", "-1"), std::invalid_argument);
  EXPECT_THROW(service.setOption("SyzygyProbeDepth", "1x"), std::invalid_argument);
  EXPECT_THROW(tablebase::Service second, std::logic_error);
}

TEST(Syzygy, FiftyMoveOutcomeMappingKeepsCursedWinsAndBlessedLossesDrawn) {
  EXPECT_EQ(tablebase::outcome(tablebase::Wdl::CursedWin), 0);
  EXPECT_EQ(tablebase::outcome(tablebase::Wdl::BlessedLoss), 0);
  EXPECT_EQ(tablebase::outcome(tablebase::Wdl::Win), 1);
  EXPECT_EQ(tablebase::outcome(tablebase::Wdl::Loss), -1);
}

TEST_F(SyzygyFiles, ProbesBothColorsAndPawnDirection) {
  struct Case {
    const char* fen;
    tablebase::Wdl wdl;
  };
  for (const Case& c : {
    Case{"7k/8/5K2/8/8/8/8/R7 w - - 0 1", tablebase::Wdl::Win},
    Case{"7k/8/5K2/8/8/8/8/R7 b - - 0 1", tablebase::Wdl::Loss},
    Case{"r7/8/8/8/8/5k2/8/7K b - - 0 1", tablebase::Wdl::Win},
    Case{"7k/8/5K2/8/8/8/8/B7 w - - 0 1", tablebase::Wdl::Draw},
    Case{"8/4P3/4K3/8/8/8/k7/8 w - - 0 1", tablebase::Wdl::Win},
    Case{"8/K7/8/8/8/4k3/4p3/8 b - - 0 1", tablebase::Wdl::Win}}) {
    SCOPED_TRACE(c.fen);
    EXPECT_EQ(service.probeWdl(BoardState(c.fen)), c.wdl);
  }
}

TEST_F(SyzygyFiles, GuardsUnsupportedStateAndProbeSettings) {
  BoardState state("7k/8/5K2/8/8/8/8/R7 w - - 1 1");
  EXPECT_FALSE(service.probeWdl(state));
  state.setPosition("7k/8/8/8/8/8/8/4K2R w K - 0 1");
  EXPECT_FALSE(service.probeWdl(state));
  EXPECT_FALSE(service.rankRoot(state, legalMoves(state), false));
  state.setPosition(fen::kStartPosition);
  EXPECT_FALSE(service.probeWdl(state));
  state.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 0 1");
  service.setOption("SyzygyProbeDepth", "3");
  EXPECT_FALSE(service.canProbeWdl(state, 2));
  EXPECT_TRUE(service.canProbeWdl(state, 3));
  service.setOption("SyzygyProbeLimit", "0");
  EXPECT_FALSE(service.enabled());
  EXPECT_FALSE(service.probeWdl(state));
}

TEST_F(SyzygyFiles, MissingDtzFallsBackAndChangingPathsReleasesOldCoverage) {
  const auto directory = std::filesystem::temp_directory_path() /
    ("kitty_syzygy_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } cleanup{directory};
  std::filesystem::create_directory(directory);
  std::filesystem::copy_file(std::filesystem::path(syzygyTestPath()) / "KRvK.rtbw", directory / "KRvK.rtbw");
  auto engine = board();
  engine.setPosition("4k3/8/8/8/8/8/8/R3K3 w - - 17 1");
  engine.setTablebaseOption("SyzygyPath", directory.string());
  EXPECT_FALSE(service.rankRoot(engine.getState(), legalMoves(engine.getState()), false));
  BoardState resetClock("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
  EXPECT_EQ(service.probeWdl(resetClock), tablebase::Wdl::Win);
  auto result = engine.search(2, std::nullopt, [](const auto&) {});
  EXPECT_TRUE(result.bestMove);
  EXPECT_LT(result.score, tablebase::kWinScore);
  engine.setTablebaseOption("SyzygyPath", (directory / "missing").string());
  EXPECT_FALSE(service.enabled());
  EXPECT_FALSE(service.probeWdl(resetClock));
  engine.setTablebaseOption("SyzygyPath", syzygyTestPath());
  EXPECT_TRUE(service.rankRoot(engine.getState(), legalMoves(engine.getState()), false));
  engine.setTablebaseOption("SyzygyPath", "<empty>");
  EXPECT_FALSE(service.enabled());
}

TEST_F(SyzygyFiles, CaptureReachesCoverageAtInclusiveSearchDepthBoundary) {
  service.setOption("SyzygyProbeLimit", "3");
  auto engine = board();
  engine.setPosition("4k3/8/8/8/8/8/p7/R3K3 w - - 0 1");
  const auto& state = engine.getState();
  EXPECT_FALSE(service.canProbeDuringSearch(state, 1));
  EXPECT_TRUE(service.canProbeDuringSearch(state, 2));
  engine.search(1, std::nullopt, [](const auto&) {});
  EXPECT_EQ(engine.tablebaseHits(), 0u);
  const auto deeper = engine.search(2, std::nullopt, [](const auto&) {});
  EXPECT_GT(engine.tablebaseHits(), 0u);
  ASSERT_TRUE(deeper.bestMove);
  EXPECT_EQ(notation::moveToString(*deeper.bestMove), "a1a2");
  EXPECT_GE(deeper.score, tablebase::kWinScore - 1);

  engine.setTablebaseOption("SyzygyProbeDepth", "0");
  EXPECT_TRUE(service.canProbeDuringSearch(state, 1));
  const auto horizon = engine.search(1, std::nullopt, [](const auto&) {});
  EXPECT_GT(engine.tablebaseHits(), 0u);
  ASSERT_TRUE(horizon.bestMove);
  EXPECT_EQ(notation::moveToString(*horizon.bestMove), "a1a2");
  EXPECT_GE(horizon.score, tablebase::kWinScore - 1);

  // Root DTZ still runs even when the WDL depth setting exceeds search depth.
  engine.setTablebaseOption("SyzygyProbeDepth", "64");
  engine.setPosition("4k3/8/8/8/8/8/8/R3K3 w - - 17 1");
  EXPECT_TRUE(service.canProbeDuringSearch(engine.getState(), 1));
  engine.search(1, std::nullopt, [](const auto&) {});
  EXPECT_EQ(engine.tablebaseHits(), 1u);
  engine.setTablebaseOption("SyzygyProbeLimit", "0");
  EXPECT_FALSE(service.canProbeDuringSearch(engine.getState(), 64));
}

TEST_F(SyzygyFiles, RootConversionMatchesLegalPromotionsAndEnPassant) {
  for (const char* fen : {
    "8/4P3/4K3/8/8/8/k7/8 w - - 0 1",
    "8/K7/8/8/8/4k3/4p3/8 b - - 0 1",
    "8/8/8/3pP3/8/4K3/8/k7 w - d6 0 1"}) {
    SCOPED_TRACE(fen);
    BoardState state(fen);
    auto moves = legalMoves(state);
    auto ranking = service.rankRoot(state, moves, false);
    ASSERT_TRUE(ranking);
    ASSERT_FALSE(ranking->bestMoves.empty());
    for (const auto& move : ranking->bestMoves) {
      EXPECT_NE(std::find(moves.begin(), moves.end(), move), moves.end());
      BoardState next = state;
      if (state.getSideToMove() == White) {
        next.makeMove<White>(move, testing_support::NoOpMoveCallback{});
      } else {
        next.makeMove<Black>(move, testing_support::NoOpMoveCallback{});
      }
      EXPECT_EQ(next.getSideToMove(), getOtherSide(state.getSideToMove()));
    }
  }
}

TEST_F(SyzygyFiles, RootRankIsPreservedAndProbedOncePerGo) {
  auto engine = board();
  engine.setPosition("4k3/8/8/8/8/8/8/R3K3 w - - 17 1");
  const BoardState before = engine.getState();
  auto ranking = service.rankRoot(before, legalMoves(before), false);
  ASSERT_TRUE(ranking);
  service.setOption("SyzygyProbeDepth", "64"); // Isolate the single root probe.
  uint64_t hits = 0;
  auto result = engine.search(4, std::nullopt, [&](const auto&) { hits += engine.tablebaseHits(); });
  ASSERT_TRUE(result.bestMove);
  EXPECT_NE(std::find(ranking->bestMoves.begin(), ranking->bestMoves.end(), *result.bestMove), ranking->bestMoves.end());
  EXPECT_EQ(hits, 1u);
  EXPECT_GE(result.score, tablebase::kWinScore);
  EXPECT_LT(result.score, -evaluation::kCheckmateScore - 100);
  EXPECT_EQ(engine.getState().getHash(), before.getHash());
  EXPECT_EQ(engine.getState().getHalfmoveClock(), before.getHalfmoveClock());
  hits = 0;
  engine.search(2, std::nullopt, [&](const auto&) { hits += engine.tablebaseHits(); });
  EXPECT_EQ(hits, 1u);
}

TEST_F(SyzygyFiles, RespectsClockBoundaryAndMatePrecedence) {
  auto engine = board();
  engine.setPosition("7k/8/5KQ1/8/8/8/8/8 w - - 99 1");
  EXPECT_EQ(engine.search(2, std::nullopt, [](const auto&) {}).score, -evaluation::kCheckmateScore - 1);
  engine.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 99 1");
  EXPECT_EQ(engine.search(2, std::nullopt, [](const auto&) {}).score, 0);
  engine.setPosition("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1");
  EXPECT_EQ(engine.search(1, std::nullopt, [](const auto&) {}).score, evaluation::kCheckmateScore);
}

TEST_F(SyzygyFiles, UciOptionsAndReconfigurationPreserveGameHistory) {
  auto engine = board();
  engine.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 0 1");
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (auto move : {"a1b1", "h8h7", "b1a1", "h7h8"}) {
      engine.playMove(move);
    }
  }
  const auto previous = engine.getState().getHash();
  engine.setTablebaseOption("SyzygyProbeDepth", "2");
  EXPECT_EQ(engine.getState().getHash(), previous);
  EXPECT_EQ(engine.search(2, std::nullopt, [](const auto&) {}).score, 0);
  std::istringstream input("uci\nsetoption name SyzygyProbeLimit value 0\nisready\ngo depth 1\nquit\n");
  std::ostringstream output, human;
  user_interface::TerminalUI ui(engine, input, output, human);
  ui.run();
  EXPECT_NE(output.str().find("option name SyzygyPath"), std::string::npos);
  EXPECT_NE(output.str().find("readyok"), std::string::npos);
  EXPECT_NE(output.str().find("tbhits 0"), std::string::npos);
  EXPECT_FALSE(service.enabled());
}

TEST_F(SyzygyFiles, RootLossDoesNotOverrideARepetitionDrawInGameHistory) {
  auto engine = board();
  engine.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 0 1");
  for (const auto move : {"a1b1", "h8h7", "b1a1", "h7h8", "a1b1", "h8h7", "b1a1"}) {
    engine.playMove(move);
  }
  EXPECT_FALSE(service.rankRoot(engine.getState(), legalMoves(engine.getState()), true));
  const auto result = engine.search(3, std::nullopt, [](const auto&) {});
  EXPECT_EQ(result.score, 0);
  ASSERT_TRUE(result.bestMove);
  EXPECT_EQ(notation::moveToString(*result.bestMove), "h7h8");
}

TEST(Syzygy, OptionParserPreservesInternalSpaces) {
  std::string parsedName, parsedValue;
  uci::TablebaseProtocol protocol{
    [&](auto name, auto value) {
      parsedName = name;
      parsedValue = value;
    },
    [] {}, [] {}, [] {}, [](auto, auto) {}, [](auto) {}, [] {}};
  protocol.send("setoption name SyzygyPath value D:\\Chess  Tables;E:\\Other Tables  ");
  EXPECT_EQ(parsedName, "SyzygyPath");
  EXPECT_EQ(parsedValue, "D:\\Chess  Tables;E:\\Other Tables");
  protocol.send("setoption name SyzygyPath value");
  EXPECT_TRUE(parsedValue.empty());
}

TEST(Syzygy, OptionParserHandlesWhitespaceAndDelegatesExistingCommands) {
  std::string parsedName, parsedValue;
  int options = 0, ready = 0;
  uci::TablebaseProtocol protocol{
    [&](auto name, auto value) {
      parsedName = name;
      parsedValue = value;
      ++options;
    },
    [] {}, [&] { ++ready; }, [] {}, [](auto, auto) {}, [](auto) {}, [] {}};
  protocol.send("  \tsetoption\tname Example  Option value  path with  spaces \t");
  EXPECT_EQ(parsedName, "Example  Option");
  EXPECT_EQ(parsedValue, "path with  spaces");
  protocol.send("setoption name Example Option");
  EXPECT_EQ(parsedName, "Example Option");
  EXPECT_TRUE(parsedValue.empty());
  protocol.send("isready");
  EXPECT_EQ(ready, 1);
  EXPECT_EQ(options, 2);
  EXPECT_THROW(protocol.send("setoption"), std::invalid_argument);
  EXPECT_THROW(protocol.send("setoption value 3"), std::invalid_argument);
  EXPECT_THROW(protocol.send("setoption name"), std::invalid_argument);
  EXPECT_EQ(options, 2);
}
