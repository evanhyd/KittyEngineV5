#include "board.h"
#include "negamax_search_policy.h"
#include "handcraft_evaluation_policy.h"
#include "equal_percentage_time_control_policy.h"
#include "search_draw.h"
#include <gtest/gtest.h>

using namespace bb;

TEST(SearchDraw, RequiresThreeOccurrencesInGameButStopsSearchCycles) {
  searching::PositionHistory history;
  for (uint64_t key : {1, 2, 3, 4, 1}) {
    history.push(key);
  }
  EXPECT_FALSE(searching::draws::isRepetition(history, 4, 0));
  EXPECT_TRUE(searching::draws::isRepetition(history, 4, 4));
  for (uint64_t key : {2, 3, 4, 1}) {
    history.push(key);
  }
  EXPECT_TRUE(searching::draws::isRepetition(history, 8, 0));
  EXPECT_FALSE(searching::draws::isRepetition(history, 0, 0));
}

TEST(SearchDraw, NormalizesOnlyUncapturableEnPassantWithoutChangingRawIdentity) {
  const BoardState noEp("4k3/8/8/3p4/8/8/8/4K3 w - - 0 1");
  const BoardState ep("4k3/8/8/3p4/8/8/8/4K3 w - d6 0 1");
  EXPECT_NE(noEp.getHash(), ep.getHash());
  EXPECT_EQ(noEp.getRepetitionHash(), ep.getRepetitionHash());
  const BoardState legal("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
  const BoardState legalWithout("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
  EXPECT_NE(legal.getRepetitionHash(), legalWithout.getRepetitionHash());
  const BoardState pinned("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
  const BoardState pinnedWithout("k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
  EXPECT_EQ(pinned.getRepetitionHash(), pinnedWithout.getRepetitionHash());
}

TEST(SearchDraw, CheckmatePrecedesFiftyMoveDrawAtRootAndInSearch) {
  Board board{searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, 80, 1024},
    time_control::EqualPercentageTimeControlPolicy{0.05f}};
  board.setPosition("7k/6Q1/5K2/8/8/8/8/8 b - - 100 1");
  const auto mate = board.search(1, std::nullopt, [](auto&) {});
  EXPECT_EQ(mate.score, evaluation::kCheckmateScore);
  EXPECT_FALSE(mate.bestMove);
  board.setPosition("7k/8/5KQ1/8/8/8/8/8 w - - 99 1");
  const auto mateInOne = board.search(1, std::nullopt, [](auto&) {});
  EXPECT_EQ(mateInOne.score, -evaluation::kCheckmateScore - 1);
  board.setPosition("7k/8/5KQ1/8/8/8/8/8 w - - 100 1");
  const auto draw = board.search(1, std::nullopt, [](auto&) {});
  EXPECT_EQ(draw.score, 0);
  EXPECT_TRUE(draw.bestMove);
}

TEST(SearchDraw, CachedRootRespectsDifferentHalfmoveClock) {
  Board board{searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, 80, 1024},
    time_control::EqualPercentageTimeControlPolicy{0.05f}};
  board.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 0 1");
  EXPECT_GT(board.search(2, std::nullopt, [](auto&) {}).score, 0);
  board.setPosition("7k/8/5K2/8/8/8/8/R7 w - - 99 1");
  EXPECT_EQ(board.search(2, std::nullopt, [](auto&) {}).score, 0);
}
