#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "nnue_evaluation_policy.h"
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

  std::filesystem::path testWeightsPath() {
    static TemporaryFile checkpoint;
    static const bool saved = [] {
      auto model = std::make_unique<nnue::NnueModel>();
      model->accumulator().biases()[0] = 0.25f;
      auto weights = model->accumulator().weights();
      for (size_t feature = 0; feature < nnue::NnueModel::kInputSize; ++feature) {
        weights[feature * nnue::NnueModel::kAccumulatorSize] =
          static_cast<float>(feature % 17) / 1000.0f;
      }
      model->firstHidden().weights()[0] = 1.0f;
      model->firstHidden().weights()[nnue::NnueModel::kAccumulatorSize] = 0.5f;
      model->secondHidden().weights()[0] = 1.0f;
      model->outputLayer().weights()[0] = 100.0f;
      nnue::NnueSerializer::save(checkpoint.path, *model);
      return true;
    }();
    (void)saved;
    return checkpoint.path;
  }

  void writeHeader(std::ostream& output, uint64_t inputSize = nnue::NnueModel::kInputSize) {
    constexpr std::array<char, 8> magic{'N', 'N', 'U', 'E', 'S', 'G', 'D', '5'};
    constexpr std::array<uint64_t, 3> progress{};
    output.write(magic.data(), magic.size());
    for (uint64_t dimension : {inputSize, uint64_t{256}, uint64_t{32}, uint64_t{32}}) {
      output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    }
    output.write(reinterpret_cast<const char*>(progress.data()), sizeof(progress));
  }

  void expectFreshScore(evaluation::NNUEEvaluationPolicy& incremental,
                        evaluation::NNUEEvaluationPolicy& fresh,
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
  accumulatorWeights[7 * nnue::NnueModel::kAccumulatorSize] = 0.5f;
  accumulatorWeights[9 * nnue::NnueModel::kAccumulatorSize] = 0.25f;
  auto firstWeights = model->firstHidden().weights();
  firstWeights[0] = 1.0f;
  model->secondHidden().weights()[0] = 1.0f;
  model->outputLayer().weights()[0] = 1.0f;

  const std::array<uint32_t, 1> whiteFeature{7};
  const std::array<uint32_t, 1> blackFeature{9};
  model->feedFull(whiteFeature, 0);
  model->feedFull(blackFeature, 1);
  EXPECT_NEAR(model->evaluate(0), 1.0f / 256.0f, 1e-7f);
  EXPECT_NEAR(model->evaluate(1), 1.0f / 65536.0f, 1e-7f);
}

TEST(Nnue, NewModelUsesSquaredClippedReluInEveryHiddenLayer) {
  auto model = std::make_unique<nnue::NnueModel>();
  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = 0.5f;
  model->firstHidden().weights()[0] = 1.0f;
  model->secondHidden().weights()[0] = 1.0f;
  model->outputLayer().weights()[0] = 1.0f;

  const std::array<uint32_t, 1> active{7};
  const std::array<uint32_t, 0> empty{};
  model->feedFull(active, 0);
  model->feedFull(empty, 1);
  EXPECT_NEAR(model->evaluate(0), 1.0f / 256.0f, 1e-7f);

  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = -0.5f;
  model->feedFull(active, 0);
  EXPECT_EQ(model->evaluate(0), 0.0f);

  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = 2.0f;
  model->firstHidden().weights()[0] = 2.0f;
  model->secondHidden().weights()[0] = 2.0f;
  model->feedFull(active, 0);
  EXPECT_EQ(model->evaluate(0), 1.0f);
}

TEST(Nnue, CheckpointRequiresSCReLUVersion) {
  EXPECT_THROW((void)nnue::NnueSerializer::load(weightsPath()), std::runtime_error);
  auto model = std::make_unique<nnue::NnueModel>();
  TemporaryFile checkpoint;
  nnue::NnueSerializer::save(checkpoint.path, *model);
  std::array<char, 8> magic{};
  std::ifstream input(checkpoint.path, std::ios::binary);
  input.read(magic.data(), magic.size());
  EXPECT_EQ(std::string_view(magic.data(), magic.size()), "NNUESGD5");

  EXPECT_NO_THROW((void)nnue::NnueSerializer::load(checkpoint.path));
}

TEST(Nnue, NewCheckpointLoadsAndReportsCorruption) {
  const auto trainedWeights = testWeightsPath();
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
  evaluation::NNUEEvaluationPolicy incremental{testWeightsPath()};
  evaluation::NNUEEvaluationPolicy fresh{testWeightsPath()};
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
  evaluation::NNUEEvaluationPolicy incremental{testWeightsPath()};
  evaluation::NNUEEvaluationPolicy fresh{testWeightsPath()};
  BoardState state{fen::kStartPosition};
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
  expectFreshScore(incremental, fresh, state);

  state.setPosition("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  incremental.reset();
  expectFreshScore(incremental, fresh, state);
}

TEST(Nnue, LongMoveSequenceWithoutEvaluationRefreshesSafely) {
  evaluation::NNUEEvaluationPolicy incremental{testWeightsPath()};
  evaluation::NNUEEvaluationPolicy fresh{testWeightsPath()};
  BoardState state{fen::kStartPosition};
  incremental.evaluate(state);

  const Move whiteOut(G1, F3, Knight);
  const Move blackOut(G8, F6, Knight);
  const Move whiteBack(F3, G1, Knight);
  const Move blackBack(F6, G8, Knight);
  for (int cycle = 0; cycle < 30; ++cycle) {
    state.makeMove<White>(whiteOut, incremental);
    state.makeMove<Black>(blackOut, incremental);
    state.makeMove<White>(whiteBack, incremental);
    state.makeMove<Black>(blackBack, incremental);
  }

  expectFreshScore(incremental, fresh, state);
}

TEST(Nnue, RepeatedSearchesAndBoardResetKeepScores) {
  Board board{
    searching::NegamaxSearchPolicy{
      evaluation::NNUEEvaluationPolicy{testWeightsPath()}, 80, 1u << 16
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

TEST(HandCraftEvaluation, BenchmarkSpeed) {
  evaluation::HandCraftEvaluationPolicy hce;
  std::vector<BoardState> positions = {
    BoardState{fen::kStartPosition},
    BoardState{fen::kKiwipete},
    BoardState{fen::kRookEndgame},
    BoardState{"r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQK2R w KQkq - 0 7"},
    BoardState{"3r2k1/5ppp/2p1p1b1/1p6/2Pq1P2/1P1PN2P/rP1Q1RP1/3R2K1 b - - 0 26"}
  };

  constexpr int iterationsPerPos = 200'000;
  volatile int32_t sink = 0;

  const auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < iterationsPerPos; ++iter) {
    for (const auto& pos : positions) {
      sink += hce.evaluate(pos);
    }
  }
  const auto end = std::chrono::high_resolution_clock::now();

  const double elapsedSec = std::chrono::duration<double>(end - start).count();
  const double totalEvals = static_cast<double>(iterationsPerPos * positions.size());
  const double evalsPerSec = totalEvals / elapsedSec;

  std::cout << "[ BENCHMARK ] HandCraftEvaluation: " << totalEvals << " evals in "
            << elapsedSec << "s (" << (evalsPerSec / 1'000'000.0) << " M evals/sec)\n";
  EXPECT_GT(evalsPerSec, 100'000.0);
}

TEST(HandCraftEvaluation, PositionScoresFidelity) {
  evaluation::HandCraftEvaluationPolicy hce;
  std::vector<std::pair<std::string_view, int32_t>> cases = {
    {fen::kStartPosition, 0},
    {fen::kKiwipete, 63},
    {fen::kRookEndgame, -55},
    {fen::kMirrorView, 134},
    {fen::kTalkChessBug, 183},
    {fen::kStevenAlt, 0},
    {"r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQK2R w KQkq - 0 7", 41},
    {"3r2k1/5ppp/2p1p1b1/1p6/2Pq1P2/1P1PN2P/rP1Q1RP1/3R2K1 b - - 0 26", -39},
    {"8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", 89},
    {"8/8/4k3/8/8/4K3/8/8 w - - 0 1", 0},
    {"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", -66}
  };

  for (auto& [fen, expected] : cases) {
    BoardState state{fen};
    int32_t score = hce.evaluate(state);
    EXPECT_EQ(score, expected) << "Mismatch on FEN: " << fen;
  }
}
