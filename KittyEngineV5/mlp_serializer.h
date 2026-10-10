#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mlp {
  // Binary file format, version 2 (all numbers are little endian):
  //   8 bytes  magic: "MLPWT001"
  //   uint32   version: 2
  //   uint32   number of layers
  //   uint64[] neuron count for each layer
  //   For each layer: float32 biases, then float32 weights for non-input layers.
  //   Weights are row major: one output neuron per row, one input per column.
  // Activation functions are supplied by the caller; they are not stored.
  template <class Model>
  class MLPSerializer {
    Model& model_;
    static constexpr std::array<char, 8> kMagic{'M', 'L', 'P', 'W', 'T', '0', '0', '1'};
    static constexpr uint32_t kVersion = 2;

    static void validate(const Model& model) {
      for (const auto& layer : model.getLayers()) {
        for (float bias : layer.getBiases()) {
          if (!std::isfinite(bias)) {
            throw std::runtime_error("MLP model has a non-finite parameter");
          }
        }
        const auto& weights = layer.getWeights();
        for (size_t row = 0; row < weights.rows(); ++row) {
          for (size_t col = 0; col < weights.cols(); ++col) {
            if (!std::isfinite(weights(row, col))) {
              throw std::runtime_error("MLP model has a non-finite parameter");
            }
          }
        }
      }
    }

    template <class UInt>
    static void writeInteger(std::ostream& stream, UInt value) {
      static_assert(std::is_unsigned_v<UInt>);
      for (unsigned shift = 0; shift < sizeof(UInt) * 8; shift += 8) {
        stream.put(static_cast<char>((value >> shift) & 0xff));
      }
      if (!stream) {
        throw std::runtime_error("Could not write MLP model");
      }
    }

    template <class UInt>
    static UInt readInteger(std::istream& stream) {
      static_assert(std::is_unsigned_v<UInt>);
      UInt value = 0;
      for (unsigned shift = 0; shift < sizeof(UInt) * 8; shift += 8) {
        const int byte = stream.get();
        if (byte == std::char_traits<char>::eof()) {
          throw std::runtime_error("MLP model is truncated");
        }
        value |= static_cast<UInt>(static_cast<unsigned char>(byte)) << shift;
      }
      return value;
    }

    static void writeFloat(std::ostream& stream, float value) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("MLP model has a non-finite parameter");
      }
      static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
      writeInteger(stream, std::bit_cast<uint32_t>(value));
    }

    static float readFloat(std::istream& stream) {
      const float value = std::bit_cast<float>(readInteger<uint32_t>(stream));
      if (!std::isfinite(value)) {
        throw std::runtime_error("MLP model has a non-finite parameter");
      }
      return value;
    }

    static void write(std::ostream& stream, const Model& model) {
      stream.write(kMagic.data(), kMagic.size());
      writeInteger(stream, kVersion);
      writeInteger(stream, static_cast<uint32_t>(model.countLayer()));
      for (const auto& layer : model.getLayers()) {
        writeInteger(stream, static_cast<uint64_t>(layer.getNumNeurons()));
      }
      for (size_t index = 0; index < model.countLayer(); ++index) {
        const auto& layer = model.getLayer(index);
        for (float bias : layer.getBiases()) {
          writeFloat(stream, bias);
        }
        if (index == 0) {
          continue;
        }
        const auto& weights = layer.getWeights();
        for (size_t row = 0; row < weights.rows(); ++row) {
          for (size_t col = 0; col < weights.cols(); ++col) {
            writeFloat(stream, weights(row, col));
          }
        }
      }
    }

    static void read(std::istream& stream, Model& model) {
      std::array<char, kMagic.size()> magic{};
      if (!stream.read(magic.data(), magic.size()) || magic != kMagic ||
          readInteger<uint32_t>(stream) != kVersion) {
        throw std::runtime_error("Invalid or unsupported MLP model file");
      }
      if (readInteger<uint32_t>(stream) != model.countLayer()) {
        throw std::runtime_error("MLP model architecture does not match");
      }
      for (const auto& layer : model.getLayers()) {
        if (readInteger<uint64_t>(stream) != layer.getNumNeurons()) {
          throw std::runtime_error("MLP model architecture does not match");
        }
      }

      for (size_t index = 0; index < model.countLayer(); ++index) {
        auto& layer = model.getLayer(index);
        for (size_t i = 0; i < layer.getNumNeurons(); ++i) {
          layer.setBias(i, readFloat(stream));
        }
        if (index == 0) {
          continue;
        }
        const auto& weights = layer.getWeights();
        for (size_t row = 0; row < weights.rows(); ++row) {
          for (size_t col = 0; col < weights.cols(); ++col) {
            layer.setWeight(row, col, readFloat(stream));
          }
        }
      }
    }

  public:
    explicit MLPSerializer(Model& model)
      : model_(model) {}

    void save(const std::filesystem::path& filename) const {
      validate(model_);
      std::ofstream stream(filename, std::ios::binary | std::ios::trunc);
      if (!stream) {
        throw std::runtime_error("Could not open MLP model file for writing");
      }
      write(stream, model_);
      stream.flush();
      if (!stream) {
        throw std::runtime_error("Could not finish writing MLP model file");
      }
      stream.close();
      if (!stream) {
        throw std::runtime_error("Could not close MLP model file");
      }
    }

    void load(const std::filesystem::path& filename) {
      std::ifstream stream(filename, std::ios::binary);
      if (!stream) {
        throw std::runtime_error("Could not read MLP model file");
      }
      Model loaded = model_;
      read(stream, loaded);
      const int trailing = stream.peek();
      if (stream.bad()) {
        throw std::runtime_error("Could not finish reading MLP model file");
      }
      if (trailing != std::char_traits<char>::eof()) {
        throw std::runtime_error("MLP model file has trailing data");
      }
      loaded.clearAccumulators();
      model_ = std::move(loaded);
    }
  };
}
