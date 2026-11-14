#pragma once
#include "nnue_mlp.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace nnue {
  class NnueSerializer {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static constexpr std::array<char, 8> kMagicV6{'N', 'N', 'U', 'E', 'S', 'G', 'D', '6'};
    static constexpr std::array<uint64_t, 5> kDimensions{
      NnueModel::kInputSize, NnueModel::kAccumulatorSize,
      NnueModel::kHidden1Size, NnueModel::kHidden2Size,
      NnueModel::kOutputBuckets
    };

    static constexpr float kQuantScaleL0 = 64.0f;
    static constexpr float kQuantScaleL1Bias = 8192.0f;
    static constexpr float kQuantScaleL1Weight = 64.0f;
    static constexpr float kQuantScaleL2Bias = 8192.0f;
    static constexpr float kQuantScaleL2Weight = 64.0f;
    static constexpr float kQuantScaleL3Bias = 2048.0f;
    static constexpr float kQuantScaleL3Weight = 16.0f;

    template <typename TargetInt, typename FloatType>
    [[nodiscard]] static constexpr TargetInt quantize(FloatType value, FloatType scale) noexcept {
      const auto quantized = static_cast<int32_t>(std::lround(value * scale));
      return static_cast<TargetInt>(std::clamp<int32_t>(
        quantized,
        static_cast<int32_t>(std::numeric_limits<TargetInt>::min()),
        static_cast<int32_t>(std::numeric_limits<TargetInt>::max())));
    }

    class Reader {
      std::ifstream stream_;
      const std::filesystem::path& filename_;
      size_t offset_ = 0;

      [[noreturn]] void fail(std::string_view section, size_t byte,
                             std::string_view reason) const {
        throw std::runtime_error(std::format("NNUE checkpoint '{}', {} at byte {}: {}",
                                             filename_.string(), section, byte, reason));
      }

    public:
      explicit Reader(const std::filesystem::path& filename)
        : stream_(filename, std::ios::binary), filename_(filename) {
        if (!stream_) {
          fail("open", 0, "could not open for reading");
        }
      }

      void bytes(void* destination, size_t count, std::string_view section) {
        const size_t start = offset_;
        stream_.read(static_cast<char*>(destination), static_cast<std::streamsize>(count));
        if (!stream_) {
          if (stream_.bad()) {
            fail(section, start, "I/O error while reading");
          }
          fail(section, start, std::format("truncated: expected {} bytes, read {}",
                                           count, stream_.gcount()));
        }
        offset_ += count;
      }

      template <typename T>
      void value(T& destination, std::string_view section) {
        bytes(&destination, sizeof(T), section);
      }

      void floats(std::span<float> destination, std::string_view section) {
        const size_t start = offset_;
        bytes(destination.data(), destination.size_bytes(), section);
        for (size_t i = 0; i < destination.size(); ++i) {
          if (!std::isfinite(destination[i])) {
            fail(section, start + i * sizeof(float), "non-finite float");
          }
        }
      }

      void finish() {
        if (stream_.peek() != std::char_traits<char>::eof()) {
          fail("end of file", offset_, "unexpected trailing data");
        }
        if (stream_.bad()) {
          fail("end of file", offset_, "I/O error while checking for trailing data");
        }
      }
    };

  public:
    static std::unique_ptr<NnueModel> load(const std::filesystem::path& filename) {
      Reader reader(filename);
      std::array<char, 8> magic{};
      reader.value(magic, "header magic");
      if (magic != kMagicV6) {
        throw std::runtime_error(std::format(
          "NNUE checkpoint '{}', header magic at byte 0: expected NNUESGD6 (Folded SCReLU 8-Bucket); older checkpoints require retraining",
          filename.string()));
      }

      for (size_t i = 0; i < kDimensions.size(); ++i) {
        uint64_t actual = 0;
        reader.value(actual, "architecture dimension");
        if (actual != kDimensions[i]) {
          throw std::runtime_error(std::format(
            "NNUE checkpoint '{}', dimension {}: expected {}, found {}",
            filename.string(), i, kDimensions[i], actual));
        }
      }

      std::array<uint64_t, 3> trainingProgress{};
      reader.value(trainingProgress, "training progress");
      auto model = std::make_unique<NnueModel>();

      // 1. Quantize L0 Biases (float * 64 -> int16_t)
      std::array<float, NnueModel::kAccumulatorSize> accumulatorBiasesRaw{};
      reader.floats(accumulatorBiasesRaw, "accumulator biases");
      auto accumulatorBiases = model->accumulator().biases();
      for (size_t i = 0; i < NnueModel::kAccumulatorSize; ++i) {
        accumulatorBiases[i] = quantize<int16_t>(accumulatorBiasesRaw[i], kQuantScaleL0);
      }

      // 2. Quantize L0 Weights (float * 64 -> int16_t, streamed row-by-row)
      std::array<float, NnueModel::kAccumulatorSize> rowFloats{};
      auto accumulatorWeights = model->accumulator().weights();
      for (size_t row = 0; row < NnueModel::kInputSize; ++row) {
        reader.floats(rowFloats, "accumulator weights");
        const size_t rowOffset = row * NnueModel::kAccumulatorSize;
        for (size_t col = 0; col < NnueModel::kAccumulatorSize; ++col) {
          accumulatorWeights[rowOffset + col] = quantize<int16_t>(rowFloats[col], kQuantScaleL0);
        }
      }

      // 3. Quantize L1 Biases (float * 8192 -> int32_t)
      std::array<float, NnueModel::kHidden1Size> firstHiddenBiasesRaw{};
      reader.floats(firstHiddenBiasesRaw, "first hidden biases");
      auto firstHiddenBiases = model->firstHidden().biases();
      for (size_t i = 0; i < NnueModel::kHidden1Size; ++i) {
        firstHiddenBiases[i] = static_cast<int32_t>(std::lround(firstHiddenBiasesRaw[i] * kQuantScaleL1Bias));
      }

      // 4. Quantize L1 Weights (float * 64 -> int8_t)
      std::vector<float> firstHiddenWeightsRaw(NnueModel::kHidden1Size * 2 * NnueModel::kAccumulatorSize);
      reader.floats(firstHiddenWeightsRaw, "first hidden weights");
      auto firstHiddenWeights = model->firstHidden().weights();
      for (size_t i = 0; i < firstHiddenWeightsRaw.size(); ++i) {
        firstHiddenWeights[i] = quantize<int8_t>(firstHiddenWeightsRaw[i], kQuantScaleL1Weight);
      }

      // 5. Quantize L2 Biases (float * 8192 -> int32_t)
      std::array<float, NnueModel::kHidden2Size> secondHiddenBiasesRaw{};
      reader.floats(secondHiddenBiasesRaw, "second hidden biases");
      auto secondHiddenBiases = model->secondHidden().biases();
      for (size_t i = 0; i < NnueModel::kHidden2Size; ++i) {
        secondHiddenBiases[i] = static_cast<int32_t>(std::lround(secondHiddenBiasesRaw[i] * kQuantScaleL2Bias));
      }

      // 6. Quantize L2 Weights (float * 64 -> int8_t)
      std::array<float, NnueModel::kHidden2Size * NnueModel::kHidden1Size> secondHiddenWeightsRaw{};
      reader.floats(secondHiddenWeightsRaw, "second hidden weights");
      auto secondHiddenWeights = model->secondHidden().weights();
      for (size_t i = 0; i < secondHiddenWeightsRaw.size(); ++i) {
        secondHiddenWeights[i] = quantize<int8_t>(secondHiddenWeightsRaw[i], kQuantScaleL2Weight);
      }

      // 7. Quantize L3 Biases (8 floats * 2048 -> int32_t)
      std::array<float, NnueModel::kOutputBuckets> outputBiasesRaw{};
      reader.floats(outputBiasesRaw, "output biases");
      auto outputBiases = model->outputLayer().biases();
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        outputBiases[b] = static_cast<int32_t>(std::lround(outputBiasesRaw[b] * kQuantScaleL3Bias));
      }

      // 8. Quantize L3 Weights (8 * 32 floats * 16 -> int16_t)
      std::array<float, NnueModel::kOutputBuckets * NnueModel::kHidden2Size> outputWeightsRaw{};
      reader.floats(outputWeightsRaw, "output weights");
      auto outputWeights = model->outputLayer().weights();
      for (size_t i = 0; i < outputWeightsRaw.size(); ++i) {
        outputWeights[i] = quantize<int16_t>(outputWeightsRaw[i], kQuantScaleL3Weight);
      }

      reader.finish();
      return model;
    }
  };
}
