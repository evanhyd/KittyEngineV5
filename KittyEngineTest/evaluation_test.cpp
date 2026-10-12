#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "mlp_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "nnue_serializer.h"
#include "notation.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <gtest/gtest.h>

using namespace bb;

namespace {
  std::filesystem::path weightsPath() {
    return std::filesystem::path(__FILE__).parent_path().parent_path()
      / "KittyEngineV5" / "weights.bin";
  }

  struct TemporaryFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("kitty_nnue_test_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");

    ~TemporaryFile() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  };

  void writeHeader(std::ostream& output, uint64_t inputSize = nnue::NnueModel::kInputSize) {
    constexpr std::array<char, 8> magic{'N', 'N', 'U', 'E', 'S', 'G', 'D', '4'};
    constexpr std::array<uint64_t, 3> progress{};
    output.write(magic.data(), magic.size());
    for (uint64_t dimension : {inputSize, uint64_t{256}, uint64_t{32}, uint64_t{32}}) {
      output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    }
    output.write(reinterpret_cast<const char*>(progress.data()), sizeof(progress));
  }

  void expectFreshScore(evaluation::MLPEvaluationPolicy& incremental,
                        evaluation::MLPEvaluationPolicy& fresh,
                        const BoardState& state) {
    fresh.reset();
    EXPECT_NEAR(incremental.evaluate(state), fresh.evaluate(state), 1);
  }
}

TEST(Nnue, KingRelativeFeaturesUseBothPerspectives) {
  const BoardState state{"4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"};
  const auto white = evaluation::NnueLayout::encode(
    state, White, evaluation::NnueLayout::kingSquare(state, White));
  const auto black = evaluation::NnueLayout::encode(
    state, Black, evaluation::NnueLayout::kingSquare(state, Black));

  ASSERT_EQ(white.size(), 1u);
  ASSERT_EQ(black.size(), 1u);
  EXPECT_EQ(white.front(), 38'452u);
  EXPECT_EQ(black.front(), 38'732u);
  EXPECT_EQ(white.front(), evaluation::NnueLayout::pieceIndex(White, White, Pawn, E1, E2));
  EXPECT_EQ(black.front(), evaluation::NnueLayout::pieceIndex(Black, White, Pawn, E8, E2));

  const BoardState differentRights{"4k3/8/8/8/8/8/4P3/4K3 w Kq e3 0 1"};
  const auto sameWhite = evaluation::NnueLayout::encode(
    differentRights, White, evaluation::NnueLayout::kingSquare(differentRights, White));
  EXPECT_EQ(sameWhite.front(), white.front());
}

TEST(Nnue, DenseInputPutsMoverBeforeOpponent) {
  auto model = std::make_unique<nnue::NnueModel>();
  auto accumulatorWeights = model->accumulator().weights();
  accumulatorWeights[7 * nnue::NnueModel::kAccumulatorSize] = 100.0f;
  accumulatorWeights[9 * nnue::NnueModel::kAccumulatorSize] = 200.0f;
  auto firstWeights = model->firstHidden().weights();
  firstWeights[0] = 1.0f;
  firstWeights[nnue::NnueModel::kAccumulatorSize] = 2.0f;
  model->secondHidden().weights()[0] = 1.0f;
  model->outputLayer().weights()[0] = 1.0f;

  const std::array<uint32_t, 1> whiteFeature{7};
  const std::array<uint32_t, 1> blackFeature{9};
  model->feedFull(whiteFeature, 0);
  model->feedFull(blackFeature, 1);
  EXPECT_NEAR(model->evaluate(0), nnue::scaledSigmoid(500.0f), 1e-6f);
  EXPECT_NEAR(model->evaluate(1), nnue::scaledSigmoid(400.0f), 1e-6f);
}

TEST(Nnue, AccumulatorAndHiddenLayersUseLeakyRelu) {
  auto model = std::make_unique<nnue::NnueModel>();
  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = -100.0f;
  model->firstHidden().weights()[0] = -1.0f;
  model->secondHidden().weights()[0] = 1.0f;
  model->outputLayer().weights()[0] = 1.0f;

  const std::array<uint32_t, 1> active{7};
  const std::array<uint32_t, 0> empty{};
  model->feedFull(active, 0);
  model->feedFull(empty, 1);
  EXPECT_NEAR(model->evaluate(0), nnue::scaledSigmoid(1.0f), 1e-6f);

  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = 1.0f;
  model->firstHidden().weights()[0] = -100.0f;
  model->secondHidden().weights()[0] = -1.0f;
  model->feedFull(active, 0);
  EXPECT_NEAR(model->evaluate(0), nnue::scaledSigmoid(1.0f), 1e-6f);
}

TEST(Nnue, NewCheckpointLoadsAndReportsCorruption) {
  const auto trainedWeights = weightsPath();
  ASSERT_TRUE(std::filesystem::exists(trainedWeights));
  EXPECT_EQ(std::filesystem::file_size(trainedWeights), 42'014'148u);
  EXPECT_NO_THROW((void)nnue::NnueSerializer::load(trainedWeights));

  TemporaryFile malformed;
  {
    std::ofstream output(malformed.path, std::ios::binary);
    writeHeader(output, nnue::NnueModel::kInputSize + 1);
  }
  try {
    (void)nnue::NnueSerializer::load(malformed.path);
    FAIL() << "Expected a dimension error";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("dimension 0"), std::string::npos);
  }

  {
    std::fstream file(malformed.path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(0);
    file.put('X');
  }
  try {
    (void)nnue::NnueSerializer::load(malformed.path);
    FAIL() << "Expected a signature error";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("header magic"), std::string::npos);
  }

  {
    std::ofstream output(malformed.path, std::ios::binary | std::ios::trunc);
    writeHeader(output);
    output.put('x');
  }
  try {
    (void)nnue::NnueSerializer::load(malformed.path);
    FAIL() << "Expected truncated accumulator biases";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("accumulator biases"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("truncated"), std::string::npos);
  }

  std::filesystem::copy_file(trainedWeights, malformed.path,
                             std::filesystem::copy_options::overwrite_existing);
  {
    std::fstream file(malformed.path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(64);
    const float invalid = std::numeric_limits<float>::infinity();
    file.write(reinterpret_cast<const char*>(&invalid), sizeof(invalid));
  }
  try {
    (void)nnue::NnueSerializer::load(malformed.path);
    FAIL() << "Expected a non-finite accumulator bias error";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("accumulator biases"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("non-finite"), std::string::npos);
  }

  {
    std::fstream file(malformed.path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(64);
    const float valid = 0.0f;
    file.write(reinterpret_cast<const char*>(&valid), sizeof(valid));
  }
  {
    std::ofstream file(malformed.path, std::ios::binary | std::ios::app);
    file.put('x');
  }
  try {
    (void)nnue::NnueSerializer::load(malformed.path);
    FAIL() << "Expected trailing data error";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("trailing data"), std::string::npos);
  }
}

TEST(Nnue, DeferredMovesAndUndosMatchFreshInference) {
  evaluation::MLPEvaluationPolicy incremental{weightsPath()};
  evaluation::MLPEvaluationPolicy fresh{weightsPath()};
  const struct Case { std::string_view fen; Move move; } cases[] = {
    {fen::kStartPosition, Move(E2, E4, Pawn, NoPiece, Move::kDoublePushFlag)},
    {"4k3/8/8/8/8/8/4p3/4K3 w - - 0 1", Move(E1, E2, King, NoPiece, Move::kCaptureFlag)},
    {"4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", Move(E5, D6, Pawn, NoPiece, Move::kEnpassantFlag)},
    {"4k3/P7/8/8/8/8/8/4K3 w - - 0 1", Move(A7, A8, Pawn, Queen)},
    {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", Move(E1, G1, King, NoPiece, Move::kCastlingFlag)},
    {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", Move(A1, A2, Rook)},
    {"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", Move(A1, A8, Rook, NoPiece, Move::kCaptureFlag)},
    {"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", Move(H8, H7, Rook)},
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.fen);
    BoardState state{testCase.fen};
    incremental.reset();
    fresh.reset();
    incremental.prepare(state);
    const int32_t rootScore = incremental.evaluate(state);
    EXPECT_NEAR(rootScore, fresh.evaluate(state), 1);

    if (state.getSideToMove() == White) {
      const MoveUndo undo = state.makeMove<White>(testCase.move, incremental);
      expectFreshScore(incremental, fresh, state);
      state.unmakeMove<White>(testCase.move, undo, incremental);
    } else {
      const MoveUndo undo = state.makeMove<Black>(testCase.move, incremental);
      expectFreshScore(incremental, fresh, state);
      state.unmakeMove<Black>(testCase.move, undo, incremental);
    }
    EXPECT_NEAR(incremental.evaluate(state), rootScore, 1);

    incremental.reset();
    if (state.getSideToMove() == White) {
      const MoveUndo undo = state.makeMove<White>(testCase.move, incremental);
      state.unmakeMove<White>(testCase.move, undo, incremental);
    } else {
      const MoveUndo undo = state.makeMove<Black>(testCase.move, incremental);
      state.unmakeMove<Black>(testCase.move, undo, incremental);
    }
    EXPECT_NEAR(incremental.evaluate(state), rootScore, 1);
  }
}

TEST(Nnue, SeveralMovesBetweenEvaluationsAndAnotherRoot) {
  evaluation::MLPEvaluationPolicy incremental{weightsPath()};
  evaluation::MLPEvaluationPolicy fresh{weightsPath()};
  BoardState state{fen::kStartPosition};
  incremental.prepare(state);
  expectFreshScore(incremental, fresh, state);

  const Move whitePawn(E2, E4, Pawn, NoPiece, Move::kDoublePushFlag);
  const Move blackPawn(E7, E5, Pawn, NoPiece, Move::kDoublePushFlag);
  const Move whiteKnight(G1, F3, Knight);
  const MoveUndo whitePawnUndo = state.makeMove<White>(whitePawn, incremental);
  const MoveUndo blackPawnUndo = state.makeMove<Black>(blackPawn, incremental);
  const MoveUndo whiteKnightUndo = state.makeMove<White>(whiteKnight, incremental);
  expectFreshScore(incremental, fresh, state);

  state.unmakeMove<White>(whiteKnight, whiteKnightUndo, incremental);
  state.unmakeMove<Black>(blackPawn, blackPawnUndo, incremental);
  state.unmakeMove<White>(whitePawn, whitePawnUndo, incremental);
  incremental.prepare(state);
  expectFreshScore(incremental, fresh, state);
  incremental.prepare(state);
  expectFreshScore(incremental, fresh, state);

  state.setPosition("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  incremental.reset();
  incremental.prepare(state);
  expectFreshScore(incremental, fresh, state);
}

TEST(Nnue, RepeatedSearchesAndBoardResetKeepScores) {
  Board board{
    searching::NegamaxSearchPolicy{
      evaluation::MLPEvaluationPolicy{weightsPath()}, 80, 1u << 16
    },
    time_control::EqualPercentageTimeControlPolicy{0.05f}
  };
  const auto first = board.search(2, std::nullopt, [](const searching::SearchResult&) {});
  const auto repeated = board.search(2, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_EQ(repeated.score, first.score);
  EXPECT_EQ(repeated.bestMove, first.bestMove);

  board.reset();
  const auto afterReset = board.search(2, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_EQ(afterReset.score, first.score);

  board.setPosition("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  const auto changed = board.search(1, std::nullopt, [](const searching::SearchResult&) {});
  EXPECT_TRUE(changed.bestMove.has_value());
}

// Expected to fail: at depth 10 the current engine still chooses 26...Bxd3?
// Enable this regression test when the search or evaluation can avoid g6d3.
#if 0
TEST(EvaluationDecision, AvoidsBishopCaptureBlunderAtDepth10) {
  Board board{
    searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, 80, 1u << 20},
    time_control::EqualPercentageTimeControlPolicy{0.05f}
  };
  board.setPosition("3r2k1/5ppp/2p1p1b1/1p6/2Pq1P2/1P1PN2P/rP1Q1RP1/3R2K1 b - - 0 26");

  const auto result = board.search(10, std::nullopt, [](const searching::SearchResult&) {});
  ASSERT_TRUE(result.bestMove.has_value());
  EXPECT_NE(notation::moveToString(*result.bestMove), "g6d3");
}
#endif
