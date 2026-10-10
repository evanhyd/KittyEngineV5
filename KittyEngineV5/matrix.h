#pragma once
#include <span>
#include <vector>

namespace mlp {
  class Matrix {
  public:
    explicit Matrix(size_t rows, size_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols), columns_(rows * cols) {}

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const float& operator()(size_t row, size_t col) const { return data_[row * cols_ + col]; }

    void set(size_t row, size_t col, float value) noexcept {
      data_[row * cols_ + col] = value;
      columns_[col * rows_ + row] = value;
    }

    std::span<const float> column(size_t col) const noexcept {
      return std::span<const float>(columns_.data() + col * rows_, rows_);
    }

  private:
    size_t rows_;
    size_t cols_;
    std::vector<float> data_;
    std::vector<float> columns_;
  };
}
