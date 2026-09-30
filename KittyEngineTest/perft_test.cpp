#include "boardstate.h"
#include "perft_driver.h"
#include "position_fens.h"
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <string_view>

using namespace bb;

namespace {

template <size_t size>
void expectDetailedCounts(std::string_view fen, const std::array<perft::Result, size>& expected) {
  static constexpr perft::Config config{false, false, true};
  const BoardState state{fen};
  for (uint32_t depth = 1; depth < size; ++depth) {
    SCOPED_TRACE("depth " + std::to_string(depth));
    const auto actual = perft::countPerft<config>(state, depth);
    EXPECT_EQ(actual.nodes, expected[depth].nodes);
    EXPECT_EQ(actual.captures, expected[depth].captures);
    EXPECT_EQ(actual.enpassants, expected[depth].enpassants);
    EXPECT_EQ(actual.castles, expected[depth].castles);
    EXPECT_EQ(actual.promotions, expected[depth].promotions);
  }
}

template <size_t size>
void expectNodeCounts(std::string_view fen, const std::array<uint64_t, size>& expected) {
  static constexpr perft::Config config{false, true, false};
  const BoardState state{fen};
  for (uint32_t depth = 1; depth < size; ++depth) {
    SCOPED_TRACE("depth " + std::to_string(depth));
    EXPECT_EQ(perft::countPerft<config>(state, depth).nodes, expected[depth]);
  }
}

} // namespace

TEST(PerftCounts, InitialPositionEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#initial-position
  expectDetailedCounts(fen::kStartPosition,
                       std::array<perft::Result, 7>{{
                         {}, {20, 0, 0, 0, 0}, {400, 0, 0, 0, 0},
                         {8902, 34, 0, 0, 0}, {197281, 1576, 0, 0, 0},
                         {4865609, 82719, 258, 0, 0}, {119060324, 2812008, 5248, 0, 0}
                       }});
}

TEST(PerftCounts, KiwipeteEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-2
  expectDetailedCounts(fen::kKiwipete,
                       std::array<perft::Result, 5>{{
                         {}, {48, 8, 0, 2, 0}, {2039, 351, 1, 91, 0},
                         {97862, 17102, 45, 3162, 0}, {4085603, 757163, 1929, 128013, 15172}
                       }});
}

TEST(PerftCounts, RookEndgameEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-3
  expectDetailedCounts(fen::kRookEndgame,
                       std::array<perft::Result, 7>{{
                         {}, {14, 1, 0, 0, 0}, {191, 14, 0, 0, 0},
                         {2812, 209, 2, 0, 0}, {43238, 3348, 123, 0, 0},
                         {674624, 52051, 1165, 0, 0}, {11030083, 940350, 33325, 0, 7552}
                       }});
}

TEST(PerftCounts, MirrorViewEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-4
  expectDetailedCounts(fen::kMirrorView,
                       std::array<perft::Result, 6>{{
                         {}, {6, 0, 0, 0, 0}, {264, 87, 0, 6, 48},
                         {9467, 1021, 4, 0, 120}, {422333, 131393, 0, 7795, 60032},
                         {15833292, 2046173, 6512, 0, 329464}
                       }});
}

TEST(PerftCounts, TalkChessBugEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-5
  expectNodeCounts(fen::kTalkChessBug,
                   std::array<uint64_t, 6>{{0, 44, 1486, 62379, 2103487, 89941194}});
}

TEST(PerftCounts, StevenAltEveryDepth) {
  // source: https://chessprogramming.org/Perft_Results#position-6
  expectNodeCounts(fen::kStevenAlt,
                   std::array<uint64_t, 6>{{0, 46, 2079, 89890, 3894594, 164075551}});
}

TEST(PerftCounts, HorizontalEnPassantPinEveryDepth) {
  expectNodeCounts("7k/3p1p2/8/r1P1K1Pr/8/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 23, 160, 3995, 26757, 712872, 5070440}});
}

TEST(PerftCounts, DiagonalEnPassantPinEveryDepth) {
  expectNodeCounts("7k/4p2q/2q5/3P1P2/4K3/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 36, 201, 6985, 42904, 1511423, 9034785}});
}

TEST(PerftCounts, HorizontalCannotEnPassantEveryDepth) {
  expectNodeCounts("7k/r2pK3/8/2P5/8/8/8/8 b - - 0 1",
                   std::array<uint64_t, 7>{{0, 14, 93, 1489, 8497, 143911, 900561}});
}

