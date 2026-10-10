#pragma once
#include "nnue.h"
#include "mlp_serializer.h"
#include <array>
#include <chrono>
#include <filesystem>

namespace bb::testing_support {
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
