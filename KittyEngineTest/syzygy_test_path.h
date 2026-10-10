#pragma once
#include <cstdlib>
#include <string>

inline std::string syzygyTestPath() {
  char* value = nullptr;
  size_t length = 0;
  _dupenv_s(&value, &length, "KITTY_SYZYGY_TEST_PATH");
  std::string path = value ? value : "";
  std::free(value);
  return path;
}
