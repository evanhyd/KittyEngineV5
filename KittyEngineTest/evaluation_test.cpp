#include "board.h"
#include "equal_percentage_time_control_policy.h"
#include "handcraft_evaluation_policy.h"
#include "negamax_search_policy.h"
#include "notation.h"
#include "mlp_evaluation_policy.h"
#include "nnue.h"
#include "mlp_serializer.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <gtest/gtest.h>
#if KITTY_ENABLE_SYZYGY
#include "syzygy_test_path.h"
#include <memory>
#endif

using namespace bb;

namespace {
  constexpr size_t kNnueInputSize = 836;
  constexpr std::array<float, 4> kCastlingWeights{30.0f, 40.0f, 50.0f, 60.0f};

  // Generated test weights keep these checks independent of the trained model.
  struct TestNnueModelFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("kitty_nnue_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");

    TestNnueModelFile() {
      mlp::NNUE model(
        {{kNnueInputSize, 256, 32, 1}}, mlp::relu, mlp::scaledSigmoid);
      for (size_t layerIndex = 0; layerIndex < model.countLayer(); ++layerIndex) {
        auto& layer = model.getLayer(layerIndex);
        for (size_t neuron = 0; neuron < layer.getNumNeurons(); ++neuron) {
          layer.setBias(neuron, 0.0f);
        }
        const auto& weights = layer.getWeights();
        for (size_t row = 0; row < weights.rows(); ++row) {
          for (size_t col = 0; col < weights.cols(); ++col) {
            layer.setWeight(row, col, 0.0f);
          }
        }
      }
      for (size_t index = 0; index < 768; ++index) {
        model.getLayer(1).setWeight(0, index, 2.0f + static_cast<float>(index % 64) * 0.5f);
      }
      for (size_t i = 0; i < 4; ++i) {
        model.getLayer(1).setWeight(0, 768 + i, kCastlingWeights[i]);
      }
      for (size_t i = 0; i < 64; ++i) {
        model.getLayer(1).setWeight(0, 772 + i, 4.0f + static_cast<float>(i));
      }
      model.getLayer(2).setWeight(0, 0, 1.0f);
      model.getLayer(3).setWeight(0, 0, 1.0f);
      mlp::MLPSerializer(model).save(path);
    }

    ~TestNnueModelFile() {
      std::error_code ignored;
      std::filesystem::remove(path, ignored);
    }
  };
}

TEST(Nnue, SingleInputAndRepeatedFullInferenceAgree) {
  mlp::NNUE model(
    {{4, 2, 1}}, mlp::relu, mlp::scaledSigmoid);
  for (size_t layerIndex = 1; layerIndex < model.countLayer(); ++layerIndex) {
    auto& layer = model.getLayer(layerIndex);
    const auto& weights = layer.getWeights();
    for (size_t row = 0; row < weights.rows(); ++row) {
      layer.setBias(row, 0.0f);
      for (size_t col = 0; col < weights.cols(); ++col) {
        layer.setWeight(row, col, 0.0f);
      }
    }
  }
  model.getLayer(1).setWeight(0, 3, 100.0f);
  model.getLayer(2).setWeight(0, 0, 1.0f);

  std::array<float, 4> input{};
  const float baseline = model.infer(input)[0];
  EXPECT_FLOAT_EQ(model.infer(input)[0], baseline);
  model.initializeAccumulator(White, input);
  model.setInput(White, 3, 1.0f);
  const float incremental = model.inferPerspective(White)[0];
  input[3] = 1.0f;
  EXPECT_FLOAT_EQ(model.infer(3, 1.0f)[0], incremental);
  EXPECT_FLOAT_EQ(model.infer(input)[0], incremental);
}

TEST(Nnue, ThirtyTwoNeuronLayerMatchesScalarForwardPass) {
  mlp::Layer previous(40);
  mlp::Layer layer(32);
  layer.connect(previous);
  for (size_t neuron = 0; neuron < 32; ++neuron) {
    layer.setBias(neuron, static_cast<float>(neuron) * 0.1f - 1.5f);
    for (size_t inputIndex = 0; inputIndex < 40; ++inputIndex) {
      layer.setWeight(neuron, inputIndex,
        static_cast<float>((neuron * 7 + inputIndex * 3) % 19) * 0.03f - 0.25f);
    }
  }

  std::array<float, 40> input{};
  for (size_t activeCount : {0u, 1u, 5u, 40u}) {
    input.fill(0.0f);
    for (size_t index = 0; index < activeCount; ++index) {
      input[index] = index % 2 == 0 ? 0.75f : -1.25f;
    }
    previous.setOutput(input);
    layer.propagateForward(previous, mlp::relu);

    for (size_t neuron = 0; neuron < 32; ++neuron) {
      float expected = layer.getBiases()[neuron];
      for (size_t index = 0; index < activeCount; ++index) {
        expected += input[index] * layer.getWeights()(neuron, index);
      }
      EXPECT_NEAR(layer.getOutput()[neuron], mlp::relu(expected), 1e-5f);
    }
  }
}

TEST(Nnue, LoadsExistingVersionTwoWeightLayout) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
    ("kitty_nnue_format_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
  const std::array<unsigned char, 84> fileBytes{
    'M', 'L', 'P', 'W', 'T', '0', '0', '1',
    2, 0, 0, 0, 3, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 128, 63, 0, 0, 0, 64,
    0, 0, 64, 64, 0, 0, 128, 64,
    0, 0, 0, 0,
    0, 0, 160, 64, 0, 0, 192, 64
  };
  {
    std::ofstream stream(path, std::ios::binary);
    ASSERT_TRUE(stream);
    stream.write(reinterpret_cast<const char*>(fileBytes.data()), fileBytes.size());
    ASSERT_TRUE(stream);
  }

  mlp::NNUE model({{2, 2, 1}}, mlp::relu, mlp::scaledSigmoid);
  const std::array<float, 2> input{1.0f, 1.0f};
  model.initializeAccumulator(White, input);
  ASSERT_TRUE(model.hasAccumulator(White));

  mlp::MLPSerializer(model).load(path);
  EXPECT_FALSE(model.hasAccumulator(White));
  EXPECT_FLOAT_EQ(model.getLayer(1).getWeights()(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(model.getLayer(1).getWeights()(1, 1), 4.0f);
  EXPECT_FLOAT_EQ(model.getLayer(2).getWeights()(0, 0), 5.0f);
  EXPECT_FLOAT_EQ(model.getLayer(2).getWeights()(0, 1), 6.0f);

  const float full = model.infer(input)[0];
  model.initializeAccumulator(White, input);
  EXPECT_FLOAT_EQ(model.inferPerspective(White)[0], full);

  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

TEST(Nnue, CastlingInputsUseSideRelativeOrder) {
  constexpr std::array<std::string_view, 4> rights{"K", "Q", "k", "q"};
  constexpr std::array<std::array<size_t, 4>, 2> expectedIndex{{
    {0, 1, 2, 3}, {2, 3, 0, 1}
  }};
  for (Side perspective = White; perspective <= Black; ++perspective) {
    const std::string side = perspective == White ? "w" : "b";
    for (size_t i = 0; i < rights.size(); ++i) {
      const BoardState state{"r3k2r/8/8/8/8/8/8/R3K2R " + side + " " + std::string(rights[i]) + " - 0 1"};
      const auto input = perspective == White
        ? evaluation::MLPEvaluationPolicy::encodePosition<White>(state)
        : evaluation::MLPEvaluationPolicy::encodePosition<Black>(state);
      for (size_t index = 0; index < 4; ++index) {
        EXPECT_FLOAT_EQ(input[768 + index], index == expectedIndex[perspective][i] ? 1.0f : 0.0f);
      }
    }
  }
}

TEST(Nnue, IncrementalMovesAndUndosMatchFreshInference) {
  const TestNnueModelFile weights;
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
    evaluation::MLPEvaluationPolicy incremental{weights.path};
    evaluation::MLPEvaluationPolicy fresh{weights.path};
    incremental.prepare(state);
    const int32_t rootScore = incremental.evaluate(state);
    fresh.reset();
    EXPECT_NEAR(rootScore, fresh.evaluate(state), 1);

    if (state.getSideToMove() == White) {
      const MoveUndo undo = state.makeMove<White>(testCase.move, incremental);
      fresh.reset();
      EXPECT_NEAR(incremental.evaluate(state), fresh.evaluate(state), 1);
      state.unmakeMove<White>(testCase.move, undo, incremental);
    } else {
      const MoveUndo undo = state.makeMove<Black>(testCase.move, incremental);
      fresh.reset();
      EXPECT_NEAR(incremental.evaluate(state), fresh.evaluate(state), 1);
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

TEST(Nnue, PreparingAnotherRootRebuildsBothPerspectives) {
  const TestNnueModelFile weights;
  evaluation::MLPEvaluationPolicy cached{weights.path};
  evaluation::MLPEvaluationPolicy fresh{weights.path};
  const BoardState first{fen::kStartPosition};
  const BoardState second{"r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"};

  cached.prepare(first);
  const int32_t firstScore = cached.evaluate(first);
  cached.prepare(first);
  EXPECT_EQ(cached.evaluate(first), firstScore);

  cached.prepare(second);
  EXPECT_NEAR(cached.evaluate(second), fresh.evaluate(second), 1);
}

#if KITTY_ENABLE_SYZYGY
namespace {
  struct ObservableNnue {
    std::shared_ptr<evaluation::MLPEvaluationPolicy> model;
    void reset() { model->reset(); }
    void prepare(const BoardState& state) { model->prepare(state); }
    int32_t evaluate(const BoardState& state) { return model->evaluate(state); }
    template <bool Add> void markPiece(Side side, Piece piece, Square square) {
      model->markPiece<Add>(side, piece, square);
    }
    void markCastle(CastlePermission rights) { model->markCastle(rights); }
    void markEnpassant(Square square) { model->markEnpassant(square); }
  };
}

TEST(Nnue, TablebaseSearchRestoresIncrementalAccumulator) {
  const auto path = syzygyTestPath();
  if (path.empty()) GTEST_SKIP() << "Set KITTY_SYZYGY_TEST_PATH for real probe tests";
  tablebase::Service tables;
  tables.setOption("SyzygyPath", path);
  const TestNnueModelFile weights;
  auto model = std::make_shared<evaluation::MLPEvaluationPolicy>(weights.path);
  evaluation::MLPEvaluationPolicy fresh{weights.path};
  Board engine{searching::NegamaxSearchPolicy{ObservableNnue{model}, 80, 1u << 14, &tables},
    time_control::EqualPercentageTimeControlPolicy{0.05f}};
  for (const auto fen : {"8/4P3/4K3/8/8/8/k7/8 w - - 0 1",
                         "8/8/8/3pP3/8/4K3/8/k7 w - d6 0 1"}) {
    engine.setPosition(fen);
    uint64_t hits = 0;
    engine.search(3, std::nullopt, [&](const auto& r) { hits += r.tablebaseHits; });
    EXPECT_GT(hits, 1u); // Exercise root ranking and interior early returns.
    fresh.reset();
    EXPECT_EQ(model->evaluate(engine.getState()), fresh.evaluate(engine.getState()));
  }
}
#endif

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
