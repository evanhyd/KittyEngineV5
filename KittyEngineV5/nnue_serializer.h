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

namespace nnue {
  class NnueSerializer {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static constexpr std::array<char, 8> kMagicV8{'N', 'N', 'U', 'E', 'S', 'G', 'D', '8'};
    static constexpr std::array<uint64_t, 5> kDimensions{
      NnueModel::kInputSize, NnueModel::kAccumulatorSize,
      NnueModel::kHidden1Size, NnueModel::kHidden2Size,
      NnueModel::kOutputBuckets
    };

    static constexpr float kQuantScaleL0 = static_cast<float>(internal::kQA);           // 255.0f
    static constexpr float kQuantScaleWeight = static_cast<float>(internal::kQB);       // 64.0f
    static constexpr float kQuantScaleBias = static_cast<float>(internal::kDenseClamp); // 8128.0f

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
      if (magic != kMagicV8) {
        throw std::runtime_error(std::format(
          "NNUE checkpoint '{}', header magic at byte 0: expected NNUESGD8 (HalfKA-16 768->[16->32->1]x8); older checkpoints require retraining",
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

      // 1. L0 Biases (768 floats * 255 -> int16_t)
      std::array<float, NnueModel::kAccumulatorSize> rowFloats{};
      reader.floats(rowFloats, "accumulator biases");
      auto l0b = model->featureTransformer().biases();
      for (size_t i = 0; i < NnueModel::kAccumulatorSize; ++i) {
        l0b[i] = quantize<int16_t>(rowFloats[i], kQuantScaleL0);
      }

      // 2. L0 Weights (12288 * 768 floats * 255 -> int16_t, streamed row-by-row)
      auto l0w = model->featureTransformer().weights();
      for (size_t row = 0; row < NnueModel::kInputSize; ++row) {
        reader.floats(rowFloats, "accumulator weights");
        const size_t rowOffset = row * NnueModel::kAccumulatorSize;
        for (size_t col = 0; col < NnueModel::kAccumulatorSize; ++col) {
          l0w[rowOffset + col] = quantize<int16_t>(rowFloats[col], kQuantScaleL0);
        }
      }

      // 3. L1 Biases (8 * 16 floats * 8128 -> int32_t)
      std::array<float, NnueModel::kHidden1Size> l1BiasRaw{};
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        reader.floats(l1BiasRaw, "first hidden biases");
        auto l1b = model->firstHidden().biases(b);
        for (size_t i = 0; i < NnueModel::kHidden1Size; ++i) {
          l1b[i] = static_cast<int32_t>(std::lround(l1BiasRaw[i] * kQuantScaleBias));
        }
      }

      // 4. L1 Weights (8 * 16 * 768 floats * 64 -> int8_t, transposed into 4-byte chunk layout)
      using L1Layer = BucketedL1Layer<NnueModel::kHidden1Size, NnueModel::kPairwiseSize, NnueModel::kOutputBuckets>;
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        auto l1w = model->firstHidden().weights(b);
        for (size_t row = 0; row < NnueModel::kHidden1Size; ++row) {
          reader.floats(rowFloats, "first hidden weights");
          for (size_t col = 0; col < NnueModel::kPairwiseSize; ++col) {
            l1w[L1Layer::weightIndex(row, col)] = quantize<int8_t>(rowFloats[col], kQuantScaleWeight);
          }
        }
      }

      // 5. L2 Biases (8 * 32 floats * 8128 -> int32_t)
      std::array<float, NnueModel::kHidden2Size> l2BiasRaw{};
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        reader.floats(l2BiasRaw, "second hidden biases");
        auto l2b = model->secondHidden().biases(b);
        for (size_t i = 0; i < NnueModel::kHidden2Size; ++i) {
          l2b[i] = static_cast<int32_t>(std::lround(l2BiasRaw[i] * kQuantScaleBias));
        }
      }

      // 6. L2 Weights (8 * 32 * 16 floats * 64 -> int16_t, transposed into pair-major layout)
      using L2Layer = BucketedL2Layer<NnueModel::kHidden2Size, NnueModel::kHidden1Size, NnueModel::kOutputBuckets>;
      std::array<float, NnueModel::kHidden1Size> l2RowRaw{};
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        auto l2w = model->secondHidden().weights(b);
        for (size_t row = 0; row < NnueModel::kHidden2Size; ++row) {
          reader.floats(l2RowRaw, "second hidden weights");
          for (size_t col = 0; col < NnueModel::kHidden1Size; ++col) {
            l2w[L2Layer::weightIndex(row, col)] = quantize<int16_t>(l2RowRaw[col], kQuantScaleWeight);
          }
        }
      }

      // 7. L3 Output Biases (8 floats in logit space * 8128 -> int32_t)
      std::array<float, NnueModel::kOutputBuckets> l3BiasRaw{};
      reader.floats(l3BiasRaw, "output biases");
      auto l3b = model->outputLayer().biases();
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        l3b[b] = static_cast<int32_t>(std::lround(l3BiasRaw[b] * kQuantScaleBias));
      }

      // 8. L3 Output Weights (8 * 32 floats in logit space * 64 -> int16_t)
      std::array<float, NnueModel::kHidden2Size> l3RowRaw{};
      for (size_t b = 0; b < NnueModel::kOutputBuckets; ++b) {
        reader.floats(l3RowRaw, "output weights");
        auto l3w = model->outputLayer().weights(b);
        for (size_t col = 0; col < NnueModel::kHidden2Size; ++col) {
          l3w[col] = quantize<int16_t>(l3RowRaw[col], kQuantScaleWeight);
        }
      }

      reader.finish();
      return model;
    }
  };
}
