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
      reader.floats(model->accumulator().biases(), "accumulator biases");
      reader.floats(model->accumulator().weights(), "accumulator weights");
      reader.floats(model->firstHidden().biases(), "first hidden biases");
      reader.floats(model->firstHidden().weights(), "first hidden weights");
      reader.floats(model->secondHidden().biases(), "second hidden biases");
      reader.floats(model->secondHidden().weights(), "second hidden weights");
      reader.floats(model->outputLayer().biases(), "output bias");
      reader.floats(model->outputLayer().weights(), "output weights");
      reader.finish();
      return model;
    }

    static void save(const std::filesystem::path& filename, NnueModel& model) {
      std::ofstream stream(filename, std::ios::binary);
      if (!stream) {
        throw std::runtime_error(std::format("Could not create NNUE checkpoint '{}'", filename.string()));
      }
      stream.write(kMagicV5.data(), kMagicV5.size());
      for (uint64_t dim : kDimensions) {
        stream.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
      }
      std::array<uint64_t, 3> progress{};
      stream.write(reinterpret_cast<const char*>(progress.data()), sizeof(progress));

      auto writeFloats = [&](auto span, std::string_view section) {
        stream.write(reinterpret_cast<const char*>(span.data()), span.size_bytes());
        if (!stream) {
          throw std::runtime_error(std::format("I/O error writing {} to '{}'", section, filename.string()));
        }
      };

      writeFloats(model.accumulator().biases(), "accumulator biases");
      writeFloats(model.accumulator().weights(), "accumulator weights");
      writeFloats(model.firstHidden().biases(), "first hidden biases");
      writeFloats(model.firstHidden().weights(), "first hidden weights");
      writeFloats(model.secondHidden().biases(), "second hidden biases");
      writeFloats(model.secondHidden().weights(), "second hidden weights");
      writeFloats(model.outputLayer().biases(), "output bias");
      writeFloats(model.outputLayer().weights(), "output weights");
      stream.flush();
      if (!stream) {
        throw std::runtime_error(std::format("Could not flush NNUE checkpoint '{}'", filename.string()));
      }
    }
  };
}
