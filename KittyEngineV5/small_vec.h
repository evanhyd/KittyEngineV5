#pragma once
#include <array>

namespace bb {
  template <typename T, size_t cap = 218>
  class SmallVec {
    std::array<T, cap> data_;
    size_t size_ = 0;

  public:
    consteval size_t capacity() const noexcept {
      return cap;
    }

    constexpr size_t size() const noexcept {
      return size_;
    }

    constexpr void resize(size_t newSize) noexcept {
      size_ = newSize;
    }

    constexpr bool empty() const noexcept {
      return size_ == 0;
    }

    constexpr auto begin() noexcept {
      return data_.begin();
    }

    constexpr auto begin() const noexcept {
      return data_.begin();
    }

    constexpr auto end() noexcept {
      return data_.begin() + size_;
    }

    constexpr auto end() const noexcept {
      return data_.begin() + size_;
    }

    constexpr T& operator[](size_t index) noexcept {
      return data_[index];
    }

    constexpr const T& operator[](size_t index) const noexcept {
      return data_[index];
    }

    constexpr void push(const T& value) noexcept {
      data_[size_++] = value;
    }

    constexpr void pop() noexcept {
      --size_;
    }
  };
}
