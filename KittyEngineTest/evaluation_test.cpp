#include "handcraft_evaluation_policy.h"
#include "position_fens.h"
#include <gtest/gtest.h>
#include <string_view>

using namespace bb;

namespace {
int32_t score(std::string_view fen) {
  BoardState state{fen};
  evaluation::HandCraftEvaluationPolicy evaluator;
  return evaluator.evaluate(state);
}
}

TEST(Evaluation, StartsBalancedAndUsesSideToMovePerspective) {
  EXPECT_EQ(score(fen::kStartPosition), 0);
  constexpr std::string_view white = "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1";
  constexpr std::string_view black = "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1";
  EXPECT_EQ(score(white), 110);
  EXPECT_EQ(score(black), -score(white));
  EXPECT_EQ(score("4k3/8/8/4p3/8/8/8/4K3 b - - 0 1"), score(white));
  EXPECT_EQ(score("7k/4p3/8/3n4/8/8/8/K7 b - - 0 1"),
            score("k7/8/8/8/3N4/8/4P3/7K w - - 0 1"));
}

TEST(Evaluation, RecognizesOnlyBasicDeadPositions) {
  EXPECT_EQ(score("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), 0);
  EXPECT_EQ(score("4k3/8/8/8/8/8/8/1N2K3 w - - 0 1"), 0);
  EXPECT_EQ(score("4k3/8/8/8/8/8/8/1B2K3 w - - 0 1"), 0);
  EXPECT_NE(score("4k3/8/8/8/8/8/8/1NN1K3 w - - 0 1"), 0);
}

TEST(Evaluation, MaterialPlacementAndMobility) {
  EXPECT_EQ(score("k7/8/8/8/3N4/8/4P3/7K w - - 0 1"), 453);
  EXPECT_EQ(score("k7/8/8/8/8/8/4P3/N6K w - - 0 1"), 363);
  EXPECT_GT(score("k7/8/8/8/3N4/8/4P3/7K w - - 0 1"),
            score("k7/8/8/8/8/8/4P3/N6K w - - 0 1"));
}

TEST(Evaluation, PawnStructureAndAdvancement) {
  EXPECT_EQ(score("k7/8/8/8/4P3/8/4P3/7K w - - 0 1"), 180);
  EXPECT_EQ(score("k7/8/8/8/4P3/8/3P4/7K w - - 0 1"), 225);
  EXPECT_EQ(score("k7/8/8/8/P1P5/8/8/7K w - - 0 1"), 212);
  EXPECT_EQ(score("k7/8/8/8/PP6/8/8/7K w - - 0 1"), 240);
  EXPECT_EQ(score("k7/8/7p/4P3/8/8/8/7K w - - 0 1"), 31);
  EXPECT_EQ(score("k7/8/5p2/4P3/8/8/8/7K w - - 0 1"), 12);
}

TEST(Evaluation, RookFilesAndBishopPair) {
  EXPECT_EQ(score("p6k/8/8/8/8/8/P7/R6K w - - 0 1"), 512);
  EXPECT_EQ(score("p6k/8/8/8/8/8/1P6/R6K w - - 0 1"), 536);
  EXPECT_EQ(score("1p5k/8/8/8/8/8/1P6/R6K w - - 0 1"), 546);
  EXPECT_EQ(score("k7/8/8/8/8/8/4P3/2BB3K w - - 0 1"), 794);
  EXPECT_EQ(score("k7/8/8/8/8/8/4P3/2BN3K w - - 0 1"), 736);
}

TEST(Evaluation, OverloadAndKingWeakness) {
  EXPECT_EQ(score("3r1r1k/8/3B1B2/8/4N3/8/8/K7 w - - 0 1"), 68);
  EXPECT_EQ(score("3r1r1k/8/3B1B2/1N6/4N3/8/8/K7 w - - 0 1"), 421);
  EXPECT_EQ(score("3r1r1k/8/3B1B2/6N1/4N3/8/8/K7 w - - 0 1"), 405);
  EXPECT_EQ(score("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1"), 0);
  EXPECT_EQ(score("r3k2r/pppppppp/8/8/3P4/8/PPP1PPPP/R3K2R w - - 0 1"), 20);
}

TEST(Evaluation, BlendsPawnAndKingTablesWithMaterialPhase) {
  EXPECT_EQ(score("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"), 110);
  EXPECT_EQ(score("r3k2r/8/8/8/4P3/8/8/R3K2R w - - 0 1"), 113);
  EXPECT_EQ(score("r2qk2r/8/8/8/4P3/8/8/R2QK2R w - - 0 1"), 117);
}

TEST(Evaluation, PenalizesMissingShelterOpenFilesAndKingZonePressure) {
  EXPECT_EQ(score("r2qk2r/pppppppp/8/8/8/8/PPPPPPPP/R2QK2R w - - 0 1"), 0);
  EXPECT_EQ(score("r2qk2r/pppppppp/8/8/4P3/8/PPPP1PPP/R2QK2R w - - 0 1"), 26);
  EXPECT_EQ(score("r2qk2r/pppppppp/8/8/8/8/PPPP1PPP/R2QK2R w - - 0 1"), -96);
  EXPECT_EQ(score("r5kq/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2Q w - - 0 1"), -28);
  EXPECT_EQ(score("4r1kq/pppp1ppp/8/8/8/8/PPPP1PPP/R3K2Q w - - 0 1"), -74);
}

TEST(Evaluation, MoveAndUnmoveRestoreScore) {
  BoardState state{fen::kStartPosition};
  evaluation::HandCraftEvaluationPolicy evaluator;
  const int32_t before = evaluator.evaluate(state);
  const Move move{E2, E4, kPawn, kNoPiece, Move::kDoublePushFlag};
  const MoveUndo undo = state.makeMove<kWhite>(move);
  EXPECT_NE(evaluator.evaluate(state), before);
  state.unmakeMove<kWhite>(move, undo);
  EXPECT_EQ(evaluator.evaluate(state), before);
}
