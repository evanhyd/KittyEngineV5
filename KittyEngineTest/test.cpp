#include "../KittyEngineV5/board.cpp"
#include "../KittyEngineV5/perft_driver.h"
#include "gtest/gtest.h"
#include <array>

// https://www.chessprogramming.org/Perft_Results
TEST(TestPerft, TestInitialFEN) {
  std::array<PerftDriver::Result, 7> results = {
    PerftDriver::Result{},
    PerftDriver::Result{20, 0, 0, 0, 0},
    PerftDriver::Result{400, 0, 0, 0, 0},
    PerftDriver::Result{8902, 34, 0, 0, 0},
    PerftDriver::Result{197281, 1576, 0, 0, 0},
    PerftDriver::Result{4865609, 82719, 258, 0, 0},
    PerftDriver::Result{119060324, 2812008, 5248, 0, 0},
  };

  PerftDriver driver{ "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<true>(depth);
    EXPECT_EQ(result.nodes, results[depth].nodes);
    EXPECT_EQ(result.captures, results[depth].captures);
    EXPECT_EQ(result.enpassants, results[depth].enpassants);
    EXPECT_EQ(result.castles, results[depth].castles);
    EXPECT_EQ(result.promotions, results[depth].promotions);
  }
}

TEST(TestPerft, TestKiwipeteFEN) {
  std::array<PerftDriver::Result, 5> results = {
    PerftDriver::Result{},
    PerftDriver::Result{48, 8, 0, 2, 0},
    PerftDriver::Result{2039, 351, 1, 91, 0},
    PerftDriver::Result{97862, 17102, 45, 3162, 0},
    PerftDriver::Result{4085603, 757163, 1929, 128013, 15172},
  };

  PerftDriver driver{ "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - " };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<true>(depth);
    EXPECT_EQ(result.nodes, results[depth].nodes);
    EXPECT_EQ(result.captures, results[depth].captures);
    EXPECT_EQ(result.enpassants, results[depth].enpassants);
    EXPECT_EQ(result.castles, results[depth].castles);
    EXPECT_EQ(result.promotions, results[depth].promotions);
  }
}

TEST(TestPerft, TestRookEndGameFEN) {
  std::array<PerftDriver::Result, 7> results = {
    PerftDriver::Result{},
    PerftDriver::Result{14, 1, 0, 0, 0},
    PerftDriver::Result{191, 14, 0, 0, 0},
    PerftDriver::Result{2812, 209, 2, 0, 0},
    PerftDriver::Result{43238, 3348, 123, 0, 0},
    PerftDriver::Result{674624, 52051, 1165, 0, 0},
    PerftDriver::Result{11030083, 940350, 33325, 0, 7552},
  };

  PerftDriver driver{ "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - " };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<true>(depth);
    EXPECT_EQ(result.nodes, results[depth].nodes);
    EXPECT_EQ(result.captures, results[depth].captures);
    EXPECT_EQ(result.enpassants, results[depth].enpassants);
    EXPECT_EQ(result.castles, results[depth].castles);
    EXPECT_EQ(result.promotions, results[depth].promotions);
  }
}

TEST(TestPerft, TestMirrorViewFEN) {
  std::array<PerftDriver::Result, 6> results = {
    PerftDriver::Result{},
    PerftDriver::Result{6, 0, 0, 0, 0},
    PerftDriver::Result{264, 87, 0, 6, 48},
    PerftDriver::Result{9467, 1021, 4, 0, 120},
    PerftDriver::Result{422333, 131393, 0, 7795, 60032},
    PerftDriver::Result{15833292, 2046173, 6512, 0, 329464},
  };

  PerftDriver driver{ "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1" };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<true>(depth);
    EXPECT_EQ(result.nodes, results[depth].nodes);
    EXPECT_EQ(result.captures, results[depth].captures);
    EXPECT_EQ(result.enpassants, results[depth].enpassants);
    EXPECT_EQ(result.castles, results[depth].castles);
    EXPECT_EQ(result.promotions, results[depth].promotions);
  }
}

TEST(TestPerft, TestTalkChessBugFEN) {
  std::array<PerftDriver::Result, 6> results = {
    PerftDriver::Result{},
    PerftDriver::Result{44, 0, 0, 0, 0},
    PerftDriver::Result{1486, 0, 0, 0, 0},
    PerftDriver::Result{62379, 0, 0, 0, 0},
    PerftDriver::Result{2103487, 0, 0, 0, 0},
    PerftDriver::Result{89941194, 0, 0, 0, 0},
  };

  PerftDriver driver{ "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8" };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<false>(depth); // Turn off verbose.
    EXPECT_EQ(result.nodes, results[depth].nodes);
  }
}

TEST(TestPerft, TestStevenAltFEN) {
  std::array<PerftDriver::Result, 6> results = {
    PerftDriver::Result{},
    PerftDriver::Result{46, 0, 0, 0, 0},
    PerftDriver::Result{2079, 0, 0, 0, 0},
    PerftDriver::Result{89890, 0, 0, 0, 0},
    PerftDriver::Result{3894594, 0, 0, 0, 0},
    PerftDriver::Result{164075551, 0, 0, 0, 0},
  };

  PerftDriver driver{ "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10" };
  for (uint32_t depth = 1; depth < results.size(); ++depth) {
    PerftDriver::Result result = driver.perft<false>(depth);
    EXPECT_EQ(result.nodes, results[depth].nodes); // Turn off verbose.
  }
}
