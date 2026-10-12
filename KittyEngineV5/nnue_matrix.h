#pragma once
#include <cstddef>
#include <array>

namespace nnue {
  template <typename T, size_t rowNum, size_t colNum>
  class Matrix {
    std::array<T, rowNum* colNum> data_;

  public:
    using type = T;

    consteval size_t rows() const noexcept {
      return rowNum;
    }

    consteval size_t cols() const noexcept {
      return colNum;
    }

    constexpr T& operator()(size_t i, size_t j) noexcept {
      return data_[i * colNum + j];
    }

    constexpr const T& operator()(size_t i, size_t j) const noexcept {
      return data_[i * colNum + j];
    }

    constexpr T* data() noexcept {
      return data_.data();
    }
    
    constexpr const T* data() const noexcept {
      return data_.data();
    }
  };
}
