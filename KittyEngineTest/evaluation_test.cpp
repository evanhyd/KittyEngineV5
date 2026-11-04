#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "nnue_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "nnue_serializer.h"
#include "notation.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
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
    constexpr std::array<char, 8> magic{'N', 'N', 'U', 'E', 'S', 'G', 'D', '5'};
    constexpr std::array<uint64_t, 3> progress{};
    output.write(magic.data(), magic.size());
    for (uint64_t dimension : {inputSize, uint64_t{256}, uint64_t{32}, uint64_t{32}}) {
      output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    }
    output.write(reinterpret_cast<const char*>(progress.data()), sizeof(progress));
  }

  void writeTestCheckpoint(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary);
    writeHeader(out);
    std::vector<float> l0b(nnue::NnueModel::kAccumulatorSize, 0.0f);
    l0b[0] = 0.25f;
    out.write(reinterpret_cast<const char*>(l0b.data()), l0b.size() * sizeof(float));

    std::vector<float> l0wRow(nnue::NnueModel::kAccumulatorSize, 0.0f);
    for (size_t f = 0; f < nnue::NnueModel::kInputSize; ++f) {
      l0wRow[0] = static_cast<float>(f % 17) / 1000.0f;
      out.write(reinterpret_cast<const char*>(l0wRow.data()), l0wRow.size() * sizeof(float));
    }

    std::vector<float> l1b(nnue::NnueModel::kHiddenSize, 0.0f);
    out.write(reinterpret_cast<const char*>(l1b.data()), l1b.size() * sizeof(float));

    std::vector<float> l1w(nnue::NnueModel::kHiddenSize * 2 * nnue::NnueModel::kAccumulatorSize, 0.0f);
    l1w[0] = 1.0f;
    l1w[nnue::NnueModel::kAccumulatorSize] = 0.5f;
    out.write(reinterpret_cast<const char*>(l1w.data()), l1w.size() * sizeof(float));

    std::vector<float> l2b(nnue::NnueModel::kHiddenSize, 0.0f);
    out.write(reinterpret_cast<const char*>(l2b.data()), l2b.size() * sizeof(float));

    std::vector<float> l2w(nnue::NnueModel::kHiddenSize * nnue::NnueModel::kHiddenSize, 0.0f);
    l2w[0] = 1.0f;
    out.write(reinterpret_cast<const char*>(l2w.data()), l2w.size() * sizeof(float));

    float l3b = 0.0f;
    out.write(reinterpret_cast<const char*>(&l3b), sizeof(float));

    std::vector<float> l3w(nnue::NnueModel::kHiddenSize, 0.0f);
    l3w[0] = 100.0f;
    out.write(reinterpret_cast<const char*>(l3w.data()), l3w.size() * sizeof(float));
  }

  std::filesystem::path testWeightsPath() {
    static TemporaryFile checkpoint;
    static const bool saved = [] {
      writeTestCheckpoint(checkpoint.path);
      return true;
    }();
    (void)saved;
    return checkpoint.path;
  }

  void expectFreshScore(evaluation::NNUEEvaluationPolicy& incremental,
                        evaluation::NNUEEvaluationPolicy& fresh,
                        const BoardState& state) {
    fresh.reset();
    EXPECT_NEAR(incremental.evaluate(state), fresh.evaluate(state), 1);
  }
}

TEST(Nnue, DenseReductionMatchesScalarReference) {
  const auto check = []<size_t N, size_t P>() {
    SCOPED_TRACE(N);
    SCOPED_TRACE(P);
    auto weights = std::make_unique<nnue::Matrix<int8_t, N, P>>();
    std::array<int32_t, N> biases{};
    uint32_t rng = 7919u + static_cast<uint32_t>(N * 31 + P);
    const auto random = [&] { return rng = rng * 1664525u + 1013904223u; };
    for (size_t row = 0; row < N; ++row) {
      biases[row] = static_cast<int32_t>(static_cast<int>(row % 7) - 2) * 200;
      for (size_t col = 0; col < P; ++col) {
        (*weights)(row, col) = static_cast<int8_t>(static_cast<int>(random() % 201) - 100);
      }
    }
    auto layer = std::make_unique<nnue::DenseLayer<N, P>>(biases, *weights);
    alignas(32) std::array<uint8_t, (P + 31) / 32 * 32> inputs{};
    for (size_t fixture = 0; fixture < 64; ++fixture) {
      SCOPED_TRACE(fixture);
      for (size_t col = 0; col < P; ++col) {
        uint8_t value = static_cast<uint8_t>(random() % 128);
        if (fixture == 0) value = 0;
        if (fixture == 1) value = static_cast<uint8_t>(std::numeric_limits<int8_t>::max());
        inputs[col] = value;
      }
      layer->feedFull(inputs);
      const auto output = layer->output();
      for (size_t row = 0; row < N; ++row) {
        int32_t sum = biases[row];
        for (size_t col = 0; col < P; ++col) {
          sum += static_cast<int32_t>(inputs[col]) * static_cast<int32_t>((*weights)(row, col));
        }
        const int32_t c = std::clamp(sum, 0, 8192);
        const int32_t v = c >> 6;
        const uint8_t expected = static_cast<uint8_t>(
          std::min<int32_t>((v * v) >> 7, std::numeric_limits<int8_t>::max()));
        EXPECT_EQ(output[row], expected) << "neuron " << row;
      }
    }
  };
  check.operator()<8, 32>();
  check.operator()<32, 512>();
  check.operator()<32, 32>();
  check.operator()<40, 32>();
  check.operator()<16, 64>();
  check.operator()<1, 32>();
}

TEST(Nnue, AccumulatorMatchesScalarReference) {
  constexpr size_t N = 256;
  constexpr size_t P = 100;
  auto weights = std::make_unique<nnue::Matrix<int16_t, P, N>>();
  std::array<int16_t, N> biases{};
  uint32_t rng = 133742u;
  const auto random = [&] { return rng = rng * 1664525u + 1013904223u; };

  for (size_t col = 0; col < N; ++col) {
    biases[col] = static_cast<int16_t>(static_cast<int>(random() % 129) - 64);
    for (size_t row = 0; row < P; ++row) {
      (*weights)(row, col) = static_cast<int16_t>(static_cast<int>(random() % 255) - 127);
    }
  }

  nnue::AccumulatorLayer<N, P> layer(biases, *weights);

  // Check feedFull and incremental against scalar reference
  for (uint32_t p = 0; p < 2; ++p) {
    std::vector<uint32_t> activeFeatures = {1, 5, 12, 27, 43, 68, 89};
    layer.feedFull(activeFeatures, p);

    std::array<int16_t, N> expectedAcc = biases;
    for (uint32_t feat : activeFeatures) {
      for (size_t c = 0; c < N; ++c) {
        expectedAcc[c] += (*weights)(feat, c);
      }
    }

    // Apply incremental add & sub
    layer.feedIncremental<true>(17, p);
    for (size_t c = 0; c < N; ++c) {
      expectedAcc[c] += (*weights)(17, c);
    }

    layer.feedIncremental<false>(5, p);
    for (size_t c = 0; c < N; ++c) {
      expectedAcc[c] -= (*weights)(5, c);
    }

    const auto outputs = layer.output();
    const auto pOutput = outputs[p];

    for (size_t c = 0; c < N; ++c) {
      const int16_t clamped = std::clamp<int16_t>(expectedAcc[c], 0, 64);
      const int32_t sq = clamped * clamped;
      const uint8_t expected = static_cast<uint8_t>(
        std::min<int32_t>(sq >> 5, std::numeric_limits<int8_t>::max()));
      EXPECT_EQ(pOutput[c], expected) << "perspective " << p << " neuron " << c;
    }
  }
}

TEST(Nnue, OutputLayerMatchesScalarReference) {
  constexpr size_t P = 32;
  auto weights = std::make_unique<nnue::Matrix<int16_t, 1, P>>();
  std::array<int32_t, 1> biases{-186204};
  uint32_t rng = 987654321u;
  const auto random = [&] { return rng = rng * 1664525u + 1013904223u; };

  for (size_t i = 0; i < P; ++i) {
    (*weights)(0, i) = static_cast<int16_t>(static_cast<int>(random() % 25000) - 12500);
  }

  nnue::OutputLayer<P, nnue::linear> layer(biases, *weights);
  alignas(32) std::array<uint8_t, P> inputs{};

  for (size_t fixture = 0; fixture < 100; ++fixture) {
    int32_t scalarSum = biases[0];
    for (size_t i = 0; i < P; ++i) {
      inputs[i] = static_cast<uint8_t>(random() % 128);
      scalarSum += static_cast<int32_t>(inputs[i]) * static_cast<int32_t>((*weights)(0, i));
    }

    layer.feedFull(inputs);
    const int32_t output = layer.output();
    const int32_t expectedCp = (scalarSum + 1024) >> 11;
    EXPECT_EQ(output, expectedCp) << "fixture " << fixture;
  }
}

TEST(Nnue, RealWeightsProduceSensibleScores) {
  const auto path = weightsPath();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "weights.bin not found at " << path;
  }

  evaluation::NNUEEvaluationPolicy policy{path};
  BoardState startState{fen::kStartPosition};
  const int32_t startScore = policy.evaluate(startState);
  // Opening position should be near equality (+- 50 cp)
  EXPECT_NEAR(startScore, 0, 50);

  // Queen advantage: White has Queen + King vs Black King
  BoardState whiteWinning{"4k3/8/8/8/8/5Q2/8/4K3 w - - 0 1"};
  policy.reset();
  const int32_t whiteScore = policy.evaluate(whiteWinning);
  EXPECT_GT(whiteScore, 1000);

  // When Black is to move in the same position, score is heavily negative
  BoardState blackLosing{"4k3/8/8/8/8/5Q2/8/4K3 b - - 0 1"};
  policy.reset();
  const int32_t blackScore = policy.evaluate(blackLosing);
  EXPECT_LT(blackScore, -1000);

  // Exact color symmetry: mirrored position with Black having Queen and to move
  BoardState blackWinning{"4K3/8/8/8/8/5q2/8/4k3 b - - 0 1"};
  policy.reset();
  const int32_t blackWinningScore = policy.evaluate(blackWinning);
  EXPECT_NEAR(whiteScore, blackWinningScore, 20);
}

TEST(Nnue, BenchmarkSpeed) {
  const auto path = weightsPath();
  if (!std::filesystem::exists(path)) {
    GTEST_SKIP() << "weights.bin not found";
  }
  evaluation::NNUEEvaluationPolicy policy{path};
  BoardState startState{fen::kStartPosition};
  constexpr size_t N = 200'000;

  const auto start = std::chrono::high_resolution_clock::now();
  int32_t score = 0;
  for (size_t i = 0; i < N; ++i) {
    score += policy.evaluate(startState);
  }
  const auto end = std::chrono::high_resolution_clock::now();
  const double seconds = std::chrono::duration<double>(end - start).count();
  const double mevalsPerSec = (static_cast<double>(N) / 1e6) / seconds;

  std::cout << "[ BENCHMARK ] NNUE Evaluation: " << N << " evals in "
            << seconds << "s (" << mevalsPerSec << " M evals/sec)\n";
  std::cout << "           Per eval: " << (seconds / N * 1e9) << " ns\n";
  EXPECT_GT(score + 100000, 0);
}


TEST(Nnue, KingRelativeFeaturesUseBothPerspectives) {
  const BoardState state{"4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"};
  const auto white = evaluation::NnueLayout::encode(
    state, White, evaluation::NnueLayout::kingSquare(state, White));
  const auto black = evaluation::NnueLayout::encode(
    state, Black, evaluation::NnueLayout::kingSquare(state, Black));

  ASSERT_EQ(white.size(), 3u);
  ASSERT_EQ(black.size(), 3u);
  EXPECT_EQ(white[0], 46'132u);
  EXPECT_EQ(black[1], 46'476u);
  EXPECT_EQ(white[0], evaluation::NnueLayout::pieceIndex(White, White, Pawn, E1, E2));
  EXPECT_EQ(black[1], evaluation::NnueLayout::pieceIndex(Black, White, Pawn, E8, E2));

  const BoardState differentRights{"4k3/8/8/8/8/8/4P3/4K3 w Kq e3 0 1"};
  const auto sameWhite = evaluation::NnueLayout::encode(
    differentRights, White, evaluation::NnueLayout::kingSquare(differentRights, White));
  ASSERT_EQ(sameWhite.size(), 3u);
  EXPECT_EQ(sameWhite[0], white[0]);
  EXPECT_EQ(sameWhite[1], white[1]);
  EXPECT_EQ(sameWhite[2], white[2]);
}

TEST(Nnue, DenseInputPutsMoverBeforeOpponent) {
  auto model = std::make_unique<nnue::NnueModel>();
  auto accumulatorWeights = model->accumulator().weights();
  accumulatorWeights[7 * nnue::NnueModel::kAccumulatorSize] = 64;
  accumulatorWeights[9 * nnue::NnueModel::kAccumulatorSize] = 32;
  auto firstWeights = model->firstHidden().weights();
  firstWeights[0] = 64;
  firstWeights[nnue::NnueModel::kAccumulatorSize] = 0;
  model->secondHidden().weights()[0] = 64;
  model->outputLayer().weights()[0] = 1000;

  const std::array<uint32_t, 1> whiteFeature{7};
  const std::array<uint32_t, 1> blackFeature{9};
  model->feedFull(whiteFeature, 0);
  model->feedFull(blackFeature, 1);
  EXPECT_GT(model->evaluate(0), model->evaluate(1));
}

TEST(Nnue, NewModelUsesSquaredClippedReluInEveryHiddenLayer) {
  auto model = std::make_unique<nnue::NnueModel>();
  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = 64;
  model->firstHidden().weights()[0] = std::numeric_limits<int8_t>::max();
  model->secondHidden().weights()[0] = std::numeric_limits<int8_t>::max();
  model->outputLayer().weights()[0] = 1000;

  const std::array<uint32_t, 1> active{7};
  const std::array<uint32_t, 0> empty{};
  model->feedFull(active, 0);
  model->feedFull(empty, 1);
  EXPECT_GT(model->evaluate(0), 0.0f);

  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = -64;
  model->feedFull(active, 0);
  EXPECT_EQ(model->evaluate(0), 0.0f);

  model->accumulator().weights()[7 * nnue::NnueModel::kAccumulatorSize] = std::numeric_limits<int8_t>::max();
  model->firstHidden().weights()[0] = std::numeric_limits<int8_t>::max();
  model->secondHidden().weights()[0] = std::numeric_limits<int8_t>::max();
  model->feedFull(active, 0);
  EXPECT_GT(model->evaluate(0), 0.0f);
}

TEST(Nnue, CheckpointRequiresSCReLUVersion) {
  TemporaryFile legacyCheckpoint;
  {
    std::ofstream out(legacyCheckpoint.path, std::ios::binary);
    constexpr std::string_view legacyMagic = "NNUESGD4";
    out.write(legacyMagic.data(), legacyMagic.size());
    std::vector<char> dummy(50'402'756 - legacyMagic.size(), 0);
    out.write(dummy.data(), dummy.size());
  }
  EXPECT_THROW((void)nnue::NnueSerializer::load(legacyCheckpoint.path), std::runtime_error);
  TemporaryFile checkpoint;
  writeTestCheckpoint(checkpoint.path);
  std::array<char, 8> magic{};
  std::ifstream input(checkpoint.path, std::ios::binary);
  input.read(magic.data(), magic.size());
  EXPECT_EQ(std::string_view(magic.data(), magic.size()), "NNUESGD5");

  EXPECT_NO_THROW((void)nnue::NnueSerializer::load(checkpoint.path));
}

TEST(Nnue, NewCheckpointLoadsAndReportsCorruption) {
  const auto trainedWeights = testWeightsPath();
  ASSERT_TRUE(std::filesystem::exists(trainedWeights));
  EXPECT_EQ(std::filesystem::file_size(trainedWeights), 50'402'756u);
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

    if (state.getSide() == White) {
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
    if (state.getSide() == White) {
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
      evaluation::NNUEEvaluationPolicy{testWeightsPath()}, searching::NegamaxSearchConfig{}, 80, 1u << 16
    },
    time_control::EqualPercentageTimeControlPolicy{0.05f}
  };
  searching::SearchResult first{};
  board.search(2, std::nullopt, [&](const searching::SearchResult& r) { first = r; });
  searching::SearchResult repeated{};
  board.search(2, std::nullopt, [&](const searching::SearchResult& r) { repeated = r; });
  EXPECT_EQ(repeated.score, first.score);
  EXPECT_EQ(repeated.move, first.move);

  board.newGame();
  searching::SearchResult afterReset{};
  board.search(2, std::nullopt, [&](const searching::SearchResult& r) { afterReset = r; });
  EXPECT_EQ(afterReset.score, first.score);

  board.setPosition("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  searching::SearchResult changed{};
  board.search(1, std::nullopt, [&](const searching::SearchResult& r) { changed = r; });
  EXPECT_TRUE(static_cast<bool>(changed.move));
}

TEST(Nnue, StateNeverCorruptedAcrossMakeUnmakeTreeAndPositionChanges) {
  evaluation::NNUEEvaluationPolicy incremental{testWeightsPath()};
  evaluation::NNUEEvaluationPolicy fresh{testWeightsPath()};

  // 1. Stress test with recursive make/unmake tree traversal on Kiwipete
  BoardState kiwipete{fen::kKiwipete};
  expectFreshScore(incremental, fresh, kiwipete);

  auto explore = [&](auto& self, BoardState& state, int depth) -> void {
    if (depth == 0) {
      expectFreshScore(incremental, fresh, state);
      return;
    }
    const Side side = state.getSide();
    MoveList moves;
    if (side == White) {
      state.generateMoves<White>(moves);
      for (size_t i = 0; i < std::min<size_t>(moves.size(), 6); ++i) {
        const MoveUndo undo = state.makeMove<White>(moves[i], incremental);
        self(self, state, depth - 1);
        state.unmakeMove<White>(moves[i], undo, incremental);
        expectFreshScore(incremental, fresh, state);
      }
    } else {
      state.generateMoves<Black>(moves);
      for (size_t i = 0; i < std::min<size_t>(moves.size(), 6); ++i) {
        const MoveUndo undo = state.makeMove<Black>(moves[i], incremental);
        self(self, state, depth - 1);
        state.unmakeMove<Black>(moves[i], undo, incremental);
        expectFreshScore(incremental, fresh, state);
      }
    }
  };

  explore(explore, kiwipete, 3);
  expectFreshScore(incremental, fresh, kiwipete);

  // 2. Position changes with board.setPosition
  Board board{
    searching::NegamaxSearchPolicy{
      evaluation::NNUEEvaluationPolicy{testWeightsPath()}, searching::NegamaxSearchConfig{}, 80, 1u << 16
    },
    time_control::EqualPercentageTimeControlPolicy{0.05f}
  };

  board.setPosition(fen::kStartPosition);
  board.search(2, std::nullopt, [](const auto&) {});

  // Same king square, completely different pieces
  board.setPosition("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  board.search(2, std::nullopt, [](const auto&) {});

  // Different king square
  board.setPosition("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
  board.search(2, std::nullopt, [](const auto&) {});

  // Back to startpos
  board.setPosition(fen::kStartPosition);
  board.search(2, std::nullopt, [](const auto&) {});
}

TEST(Nnue, SearchRootInCheckInitializesAccumulatorSafely) {
  evaluation::NNUEEvaluationPolicy fresh{testWeightsPath()};
  evaluation::NNUEEvaluationPolicy searchPolicy{testWeightsPath()};

  Board board{
    searching::NegamaxSearchPolicy{
      std::move(searchPolicy), searching::NegamaxSearchConfig{}, 80, 1u << 16
    },
    time_control::EqualPercentageTimeControlPolicy{0.05f}
  };

  // Position where White is in check from Black Queen on e2.
  const std::string_view fenInCheck = "rnb1kbnr/pppp1ppp/8/8/8/8/PPPPqPPP/RNBQKBNR w KQkq - 0 1";
  board.setPosition(fenInCheck);

  searching::SearchResult result{};
  board.search(3, std::nullopt, [&](const searching::SearchResult& r) { result = r; });
  EXPECT_TRUE(static_cast<bool>(result.move));

  const std::string bestMoveStr = notation::moveToString(result.move);
  EXPECT_TRUE(bestMoveStr == "d1e2" || bestMoveStr == "f1e2" || bestMoveStr == "g1e2");

  // Subsequent search and evaluation from a normal position must remain intact.
  board.setPosition(fen::kStartPosition);
  searching::SearchResult startResult{};
  board.search(2, std::nullopt, [&](const searching::SearchResult& r) { startResult = r; });
  EXPECT_TRUE(static_cast<bool>(startResult.move));

  BoardState startState{fen::kStartPosition};
  const int32_t freshScore = fresh.evaluate(startState);
  EXPECT_GT(freshScore, -100);
  EXPECT_LT(freshScore, 100);
}


// Expected to fail: at depth 10 the current engine still chooses 26...Bxd3?
// Enable this regression test when the search or evaluation can avoid g6d3.
#if 0
TEST(EvaluationDecision, AvoidsBishopCaptureBlunderAtDepth10) {
  Board board{
    searching::NegamaxSearchPolicy{evaluation::HandCraftEvaluationPolicy{}, searching::NegamaxSearchConfig{}, 80, 1u << 20},
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


TEST(TranspositionTable, PVEntriesAreProtectedFromNonPVEviction) {
  constexpr size_t capacity = 1024;
  bb::TranspositionTable table(capacity);

  const bb::ZobristHash::Hash key1 = 0x12345678ULL;
  // key2 and key3 map to the exact same bucket as key1
  const bb::ZobristHash::Hash key2 = key1 + capacity;
  const bb::ZobristHash::Hash key3 = key1 + 2 * capacity;

  const bb::Move pvMove(E2, E4, Pawn);
  const bb::Move nonPvMove(D2, D4, Pawn);

  // 1. Store a PV entry at depth 10
  table.put(bb::TranspositionTable::Entry{
    .key = key1,
    .score = 50,
    .staticScore = 40,
    .bestMove = pvMove,
    .depth = 10,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::Exact,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = true,
  });

  const auto* e1 = table.get(key1);
  ASSERT_NE(e1, nullptr);
  EXPECT_TRUE(e1->isPV);
  EXPECT_EQ(e1->bestMove, pvMove);

  // 2. On collision with key2: key2 uses the adjacent slot in the bucket!
  table.put(bb::TranspositionTable::Entry{
    .key = key2,
    .score = -30,
    .staticScore = -20,
    .bestMove = nonPvMove,
    .depth = 12,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });

  // Both key1 (slot 0) and key2 (slot 1) are retained in the same bucket
  EXPECT_NE(table.get(key1), nullptr);
  EXPECT_NE(table.get(key2), nullptr);

  // 3. A 3rd collision key3 with shallow depth 11 cannot evict key1 (effective 14) or key2 (effective 12)
  table.put(bb::TranspositionTable::Entry{
    .key = key3,
    .score = -40,
    .staticScore = -30,
    .bestMove = nonPvMove,
    .depth = 11,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });
  EXPECT_NE(table.get(key1), nullptr);
  EXPECT_NE(table.get(key2), nullptr);
  EXPECT_EQ(table.get(key3), nullptr);

  // 4. A collision at depth 13 can evict key2 (depth 12), while key1 (PV, effective depth 14) remains protected
  table.put(bb::TranspositionTable::Entry{
    .key = key3,
    .score = -40,
    .staticScore = -30,
    .bestMove = nonPvMove,
    .depth = 13,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });
  EXPECT_NE(table.get(key1), nullptr);
  EXPECT_EQ(table.get(key2), nullptr);
  EXPECT_NE(table.get(key3), nullptr);

  // 5. A non-PV collision key2 at depth 15 evicts key3 (effective 13 < 14)
  table.put(bb::TranspositionTable::Entry{
    .key = key2,
    .score = -30,
    .staticScore = -20,
    .bestMove = nonPvMove,
    .depth = 15,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });
  EXPECT_NE(table.get(key1), nullptr);
  EXPECT_NE(table.get(key2), nullptr);
  EXPECT_EQ(table.get(key3), nullptr);

  // 6. With key1 (effective 14) and key2 (effective 15) in the bucket,
  // key1 is now the weakest slot. An insertion at depth 15 finally evicts PV key1!
  table.put(bb::TranspositionTable::Entry{
    .key = key3,
    .score = -40,
    .staticScore = -30,
    .bestMove = nonPvMove,
    .depth = 15,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });

  EXPECT_EQ(table.get(key1), nullptr);
  EXPECT_NE(table.get(key2), nullptr);
  EXPECT_NE(table.get(key3), nullptr);
}

TEST(TranspositionTable, BucketAlignmentAndExactSingleStorage) {
  static_assert(sizeof(bb::TranspositionTable::TTBucket) == std::hardware_destructive_interference_size);
  static_assert(alignof(bb::TranspositionTable::TTBucket) == std::hardware_destructive_interference_size);

  constexpr size_t capacity = 1024;
  bb::TranspositionTable table(capacity);

  const bb::ZobristHash::Hash key = 0xabcdef12ULL;
  const bb::Move move1(E2, E4, Pawn);
  const bb::Move move2(E7, E5, Pawn);

  table.put(bb::TranspositionTable::Entry{
    .key = key,
    .score = 10,
    .staticScore = 10,
    .bestMove = move1,
    .depth = 4,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::Exact,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });

  // Update same key at depth 6
  table.put(bb::TranspositionTable::Entry{
    .key = key,
    .score = 20,
    .staticScore = 10,
    .bestMove = move2,
    .depth = 6,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::Exact,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });

  const auto* e = table.get(key);
  ASSERT_NE(e, nullptr);
  EXPECT_EQ(e->depth, 6);
  EXPECT_EQ(e->bestMove, move2);
}

TEST(TranspositionTable, SameKeyPreservesPVMoveAndStatusAgainstNonPVBounds) {
  constexpr size_t capacity = 1024;
  bb::TranspositionTable table(capacity);

  const bb::ZobristHash::Hash key = 0x999999ULL;
  const bb::Move pvMove(G1, F3, Knight);
  const bb::Move scoutMove(B1, C3, Knight);

  // Store initial exact PV entry at depth 8
  table.put(bb::TranspositionTable::Entry{
    .key = key,
    .score = 25,
    .staticScore = 20,
    .bestMove = pvMove,
    .depth = 8,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::Exact,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = true,
  });

  // A non-PV scout LowerBound at depth 8 for the same key must NOT overwrite the Exact PV evaluation
  table.put(bb::TranspositionTable::Entry{
    .key = key,
    .score = 30,
    .staticScore = 20,
    .bestMove = scoutMove,
    .depth = 8,
    .age = 0,
    .scoreType = bb::TranspositionTable::ScoreType::LowerBound,
    .hasStaticScore = true,
    .inCheck = false,
    .hasInCheck = true,
    .isPV = false,
  });

  const auto* entry = table.get(key);
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->isPV);
  EXPECT_EQ(entry->scoreType, bb::TranspositionTable::ScoreType::Exact);
  EXPECT_EQ(entry->bestMove, pvMove);
}

namespace {
  searching::HistoryHeuristics makeTestHistory() {
    const searching::NegamaxSearchConfig config{};
    return searching::HistoryHeuristics(config.maxQuietHistory,
                                        config.maxRefutationHistory,
                                        config.maxFollowupHistory,
                                        config.maxCaptureHistory,
                                        config.lateMoveReductionHistoryK,
                                        config.lateMoveReductionHistoryB);
  }
}

TEST(StagedMovePicker, PreservesMoveCountAndIntegrityAcrossStages) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  BoardState state;
  state.setPosition(fen::kKiwipete);

  searching::HistoryHeuristics history = makeTestHistory();
  searching::StagedMovePicker picker(0, {}, {}, history);
  state.generateMoves<White>(picker.getMoveBuffer());
  ASSERT_GT(picker.size(), 0);

  const MoveList originalMoves = picker.getMoveBuffer();

  MoveList pickedMoves;
  for (size_t i = 0; i < originalMoves.size(); ++i) {
    const auto [m, heuristicScore] = picker.pick<kMeta>(Move{}, state);
    ASSERT_TRUE(static_cast<bool>(m));
    pickedMoves.push(m);
  }

  EXPECT_EQ(pickedMoves.size(), originalMoves.size());

  for (const auto orig : originalMoves) {
    size_t count = 0;
    for (const auto picked : pickedMoves) {
      if (picked == orig) ++count;
    }
    EXPECT_EQ(count, 1) << "Move not found or duplicated";
  }
}

TEST(StagedMovePicker, YieldsTTMoveFirstWithCorrectSEE) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  BoardState state;
  state.setPosition(fen::kKiwipete);

  searching::HistoryHeuristics history = makeTestHistory();
  searching::StagedMovePicker picker(0, {}, {}, history);
  state.generateMoves<White>(picker.getMoveBuffer());

  Move ttMove{};
  for (const auto m : picker.getMoveBuffer()) {
    if (m.isCapture()) {
      ttMove = m;
      break;
    }
  }
  ASSERT_TRUE(static_cast<bool>(ttMove));

  const int32_t expectedSee = searching::StagedMovePicker::staticExchangeEvaluation<White>(state, ttMove);

  const auto [m, heuristicScore] = picker.pick<kMeta>(ttMove, state);
  EXPECT_EQ(m, ttMove);
  EXPECT_EQ(heuristicScore, expectedSee);
}

TEST(StagedMovePicker, CachesExactSEEForViolentMoves) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  BoardState state;
  state.setPosition(fen::kKiwipete);

  searching::HistoryHeuristics history = makeTestHistory();
  searching::StagedMovePicker picker(0, {}, {}, history);
  state.generateMoves<White>(picker.getMoveBuffer());
  const size_t totalMoves = picker.size();

  for (size_t i = 0; i < totalMoves; ++i) {
    const auto [m, heuristicScore] = picker.pick<kMeta>(Move{}, state);
    if (m.isViolentMove()) {
      const int32_t directSee = searching::StagedMovePicker::staticExchangeEvaluation<White>(state, m);
      EXPECT_EQ(heuristicScore, directSee);
    } else {
      EXPECT_EQ(heuristicScore, history.getQuietScore<White>(m, {}, {}));
    }
  }
}

TEST(StagedMovePicker, StrictStageOrderMaintained) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  const std::vector<std::string_view> fens = {
    fen::kStartPosition,
    fen::kKiwipete,
    fen::kMirrorView,
    fen::kRookEndgame,
    fen::kTalkChessBug
  };

  searching::HistoryHeuristics history = makeTestHistory();

  for (const auto& fenStr : fens) {
    BoardState state;
    state.setPosition(fenStr);

    searching::StagedMovePicker picker(0, {}, {}, history);
    state.generateMoves<White>(picker.getMoveBuffer());
    const size_t totalMoves = picker.size();
    if (totalMoves == 0) continue;

    int lastStage = 0; // 1: GoodViolent, 2: Quiet/Killer, 3: BadViolent
    for (size_t i = 0; i < totalMoves; ++i) {
      const auto [m, heuristicScore] = picker.pick<kMeta>(Move{}, state);
      ASSERT_TRUE(static_cast<bool>(m));

      int stage = 0;
      if (m.isViolentMove()) {
        const int32_t realSee = searching::StagedMovePicker::staticExchangeEvaluation<White>(state, m);
        const Piece promo = m.getPromotedPieceType();
        const bool isGood = (promo == Queen) || (promo == NoPiece && realSee >= 0);
        stage = isGood ? 1 : 3;
        EXPECT_EQ(heuristicScore, realSee) << "Violent move SEE mismatch for move " << i << " in FEN: " << fenStr;
      } else {
        stage = 2;
        EXPECT_EQ(heuristicScore, history.getQuietScore<White>(m, {}, {}));
      }

      if (stage == 3) {
        EXPECT_TRUE(picker.isBadViolentMove()) << "Bad violent move not reported as bad violent in FEN: " << fenStr;
      } else {
        EXPECT_FALSE(picker.isBadViolentMove()) << "Non-bad-violent move reported as bad violent in FEN: " << fenStr;
      }

      EXPECT_LE(lastStage, stage) << "Stage ordering inverted: stage " << stage << " after stage " << lastStage << " in FEN: " << fenStr;
      lastStage = stage;
    }
  }
}

TEST(StagedMovePicker, BadViolentAtEndOfListIsNotClassifiedAsQuiet) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  BoardState state;
  // Position where White Queen on d1 can capture on d7 with negative SEE, and White also has quiet moves
  state.setPosition(fen::kKiwipete);

  searching::HistoryHeuristics history = makeTestHistory();
  searching::StagedMovePicker picker(0, {}, {}, history);
  state.generateMoves<White>(picker.getMoveBuffer());

  MoveList& moves = picker.getMoveBuffer();
  // Find a quiet move and a bad capture (SEE < 0)
  std::optional<Move> quietMove;
  std::optional<Move> badCapture;

  for (const auto m : moves) {
    if (m.isViolentMove()) {
      if (searching::StagedMovePicker::staticExchangeEvaluation<White>(state, m) < 0) {
        badCapture = m;
      }
    } else if (!quietMove) {
      quietMove = m;
    }
  }

  ASSERT_TRUE(quietMove.has_value());
  ASSERT_TRUE(badCapture.has_value());

  // Set up picker with exactly [quietMove, badCapture]
  picker.getMoveBuffer().clear();
  picker.getMoveBuffer().push(*quietMove);
  picker.getMoveBuffer().push(*badCapture);

  // First pick must be the quiet move (since there are no good violent moves)
  const auto pick1 = picker.pick<kMeta>(Move{}, state);
  EXPECT_EQ(pick1.move, *quietMove);
  EXPECT_FALSE(picker.isBadViolentMove());

  // Second pick must be the bad capture!
  const auto pick2 = picker.pick<kMeta>(Move{}, state);
  EXPECT_EQ(pick2.move, *badCapture);
  EXPECT_TRUE(picker.isBadViolentMove()) << "BUG CONFIRMED: Bad capture at end was NOT flagged as bad violent move!";
  EXPECT_EQ(pick2.heuristicScore, searching::StagedMovePicker::staticExchangeEvaluation<White>(state, *badCapture))
    << "BUG CONFIRMED: Bad capture at end did NOT receive cached SEE!";
}

TEST(StagedMovePicker, RePickRestoresStageAndPointersWithoutLosingMoves) {
  constexpr searching::NodeMeta kMeta{White, searching::NodeMeta::NodeType::Internal, true, false};
  BoardState state;
  state.setPosition(fen::kKiwipete);

  searching::HistoryHeuristics history = makeTestHistory();
  searching::StagedMovePicker picker(0, {}, {}, history);
  state.generateMoves<White>(picker.getMoveBuffer());
  const size_t totalMoves = picker.size();
  ASSERT_GT(totalMoves, 0);

  MoveList firstPass;
  for (size_t i = 0; i < totalMoves; ++i) {
    const auto [m, heuristicScore] = picker.pick<kMeta>(Move{}, state);
    firstPass.push(m);
  }
  EXPECT_EQ(firstPass.size(), totalMoves);

  // Call rePick() and pick again.
  picker.rePick();

  MoveList secondPass;
  for (size_t i = 0; i < totalMoves; ++i) {
    const auto [m, heuristicScore] = picker.pick<kMeta>(Move{}, state);
    secondPass.push(m);
  }
  EXPECT_EQ(secondPass.size(), totalMoves);
  for (const auto orig : firstPass) {
    size_t count = 0;
    for (const auto picked : secondPass) {
      if (picked == orig) ++count;
    }
    EXPECT_EQ(count, 1) << "Move not found or duplicated after rePick: " << notation::moveToString(orig);
  }
}

TEST(SearchBenchmark, DISABLED_CompareLmrAndHistoryConfigs) {
  struct BenchmarkConfig {
    std::string name;
    bb::searching::NegamaxSearchConfig config;
  };

  const std::vector<BenchmarkConfig> configs = {
    {"Baseline (Fut=100/d3, SEE=-100/d3)", bb::searching::NegamaxSearchConfig{}}
  };

  struct PositionSpec {
    std::string name;
    std::string_view fen;
    int depth;
  };

  const std::vector<PositionSpec> positions = {
    {"Startpos20", fen::kStartPosition, 20},
    {"Kiwipete16", fen::kKiwipete, 16},
    {"RookEndgame20", fen::kRookEndgame, 20}
  };

  std::cout << "\n==================================== PRUNING TUNING SCREENING BENCHMARK ====================================\n";
  std::cout << std::format("{:<45} | {:<10} | {:<10} | {:<10} | {:<12} | {:<8}\n",
                           "Configuration", "Start16", "Kiwi13", "Endgame16", "TotalNodes", "Rel %");
  std::cout << "--------------------------------------------------------------------------------------------------------------\n" << std::flush;

  uint64_t baselineTotalNodes = 0;

  for (size_t configIndex = 0; configIndex < configs.size(); ++configIndex) {
    const auto& [name, config] = configs[configIndex];
    std::cout << "\nRunning: " << name << " ...\n" << std::flush;
    uint64_t configTotalNodes = 0;
    std::vector<uint64_t> posNodes;
    std::vector<std::string> bestMoves;
    std::vector<int32_t> posScores;
    std::vector<double> elapsedTimesMs;

    for (const auto& pos : positions) {
      bb::Board board{
        bb::searching::NegamaxSearchPolicy{
          bb::evaluation::NNUEEvaluationPolicy{testWeightsPath()},
          config,
          60,
          1u << 22
        },
        bb::time_control::EqualPercentageTimeControlPolicy{0.05f}
      };

      board.setPosition(pos.fen);
      const auto start = std::chrono::high_resolution_clock::now();
      bb::searching::SearchResult result{};
      board.search(pos.depth, std::nullopt, [&](const bb::searching::SearchResult& r) { result = r; });
      const auto end = std::chrono::high_resolution_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(end - start).count();

      posNodes.push_back(result.nodes);
      configTotalNodes += result.nodes;
      bestMoves.push_back(result.move ? notation::moveToString(result.move) : "none");
      posScores.push_back(result.score);
      elapsedTimesMs.push_back(ms);
      std::cout << "  -> " << pos.name << ": " << result.nodes << " nodes, "
                << (result.move ? notation::moveToString(result.move) : "none")
                << " (" << (result.score >= 0 ? "+" : "") << result.score << "), "
                << (ms / 1000.0) << "s\n" << std::flush;
    }

    if (configIndex == 0) {
      baselineTotalNodes = configTotalNodes;
    }

    const double relPercent = baselineTotalNodes > 0
      ? (100.0 * static_cast<double>(configTotalNodes) / static_cast<double>(baselineTotalNodes))
      : 100.0;

    std::cout << std::format("{:<45} | {:<10} | {:<10} | {:<10} | {:<12} | {:>6.2f}%\n",
                             name, posNodes[0], posNodes[1], posNodes[2], configTotalNodes, relPercent);
    std::cout << std::format("   summary: [Start16: {} ({:+d}, {:.1f}s), Kiwi13: {} ({:+d}, {:.1f}s), End16: {} ({:+d}, {:.1f}s)]\n",
                             bestMoves[0], posScores[0], elapsedTimesMs[0] / 1000.0,
                             bestMoves[1], posScores[1], elapsedTimesMs[1] / 1000.0,
                             bestMoves[2], posScores[2], elapsedTimesMs[2] / 1000.0) << std::flush;
  }
  std::cout << "==============================================================================================================\n\n" << std::flush;
}
