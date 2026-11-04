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
    static constexpr std::array<char, 8> kMagicV5{'N', 'N', 'U', 'E', 'S', 'G', 'D', '5'};
    static constexpr std::array<uint64_t, 4> kDimensions{
      NnueModel::kInputSize, NnueModel::kAccumulatorSize,
      NnueModel::kHiddenSize, NnueModel::kHiddenSize
    };

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
      if (magic != kMagicV5) {
        throw std::runtime_error(std::format(
          "NNUE checkpoint '{}', header magic at byte 0: expected NNUESGD5 (SCReLU); older checkpoints require retraining",
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
      std::array<float, NnueModel::kAccumulatorSize> l0bFloats{};
      reader.floats(l0bFloats, "accumulator biases");
      auto l0bSpan = model->accumulator().biases();
      for (size_t i = 0; i < NnueModel::kAccumulatorSize; ++i) {
        const auto val = static_cast<int32_t>(std::lround(l0bFloats[i] * 64.0f));
        l0bSpan[i] = static_cast<int16_t>(std::clamp<int32_t>(
          val, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()));
      }

      // 2. Quantize L0 Weights (float * 64 -> int16_t, streamed row-by-row)
      std::array<float, NnueModel::kAccumulatorSize> rowFloats{};
      auto l0wSpan = model->accumulator().weights();
      for (size_t r = 0; r < NnueModel::kInputSize; ++r) {
        reader.floats(rowFloats, "accumulator weights");
        const size_t rowOffset = r * NnueModel::kAccumulatorSize;
        for (size_t c = 0; c < NnueModel::kAccumulatorSize; ++c) {
          const auto val = static_cast<int32_t>(std::lround(rowFloats[c] * 64.0f));
          l0wSpan[rowOffset + c] = static_cast<int16_t>(std::clamp<int32_t>(
            val, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()));
        }
      }

      // 3. Quantize L1 Biases (float * 8192 -> int32_t)
      std::array<float, NnueModel::kHiddenSize> l1bFloats{};
      reader.floats(l1bFloats, "first hidden biases");
      auto l1bSpan = model->firstHidden().biases();
      for (size_t i = 0; i < NnueModel::kHiddenSize; ++i) {
        l1bSpan[i] = static_cast<int32_t>(std::lround(l1bFloats[i] * 8192.0f));
      }

      // 4. Quantize L1 Weights (float * 64 -> int8_t)
      std::vector<float> l1wFloats(NnueModel::kHiddenSize * 2 * NnueModel::kAccumulatorSize);
      reader.floats(l1wFloats, "first hidden weights");
      auto l1wSpan = model->firstHidden().weights();
      for (size_t i = 0; i < l1wFloats.size(); ++i) {
        const auto val = static_cast<int32_t>(std::lround(l1wFloats[i] * 64.0f));
        l1wSpan[i] = static_cast<int8_t>(std::clamp<int32_t>(
          val, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()));
      }

      // 5. Quantize L2 Biases (float * 8192 -> int32_t)
      std::array<float, NnueModel::kHiddenSize> l2bFloats{};
      reader.floats(l2bFloats, "second hidden biases");
      auto l2bSpan = model->secondHidden().biases();
      for (size_t i = 0; i < NnueModel::kHiddenSize; ++i) {
        l2bSpan[i] = static_cast<int32_t>(std::lround(l2bFloats[i] * 8192.0f));
      }

      // 6. Quantize L2 Weights (float * 64 -> int8_t)
      std::array<float, NnueModel::kHiddenSize * NnueModel::kHiddenSize> l2wFloats{};
      reader.floats(l2wFloats, "second hidden weights");
      auto l2wSpan = model->secondHidden().weights();
      for (size_t i = 0; i < l2wFloats.size(); ++i) {
        const auto val = static_cast<int32_t>(std::lround(l2wFloats[i] * 64.0f));
        l2wSpan[i] = static_cast<int8_t>(std::clamp<int32_t>(
          val, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()));
      }

      // 7. Quantize L3 Bias (float * 2048 -> int32_t)
      std::array<float, 1> l3bFloat{};
      reader.floats(l3bFloat, "output bias");
      model->outputLayer().biases()[0] = static_cast<int32_t>(
        std::lround(l3bFloat[0] * 2048.0f));

      // 8. Quantize L3 Weights (float * 16 -> int16_t)
      std::array<float, NnueModel::kHiddenSize> l3wFloats{};
      reader.floats(l3wFloats, "output weights");
      auto l3wSpan = model->outputLayer().weights();
      for (size_t i = 0; i < NnueModel::kHiddenSize; ++i) {
        const auto val = static_cast<int32_t>(std::lround(l3wFloats[i] * 16.0f));
        l3wSpan[i] = static_cast<int16_t>(std::clamp<int32_t>(
          val, std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max()));
      }

      reader.finish();
      return model;
    }
  };
}
