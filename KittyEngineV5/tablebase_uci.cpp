#include "tablebase_uci.h"
#include <stdexcept>

namespace bb::uci {
  namespace {
    std::string_view takeWord(std::string_view& line) {
      constexpr std::string_view kWhitespace = " \t\r\n\f\v";
      const auto first = line.find_first_not_of(kWhitespace);
      if (first == std::string_view::npos) {
        line = {};
        return {};
      }
      line.remove_prefix(first);
      const auto end = line.find_first_of(kWhitespace);
      const auto word = line.substr(0, end);
      line.remove_prefix(word.size());
      return word;
    }
  }

  void TablebaseProtocol::send(std::string_view line) {
    auto remaining = line;
    if (takeWord(remaining) != "setoption") {
      UciProtocol::send(line);
      return;
    }
    if (takeWord(remaining) != "name") {
      throw std::invalid_argument("setoption needs name");
    }
    const auto first = takeWord(remaining);
    if (first.empty()) {
      throw std::invalid_argument("setoption needs name");
    }
    auto last = first;
    std::string_view value;
    for (auto word = takeWord(remaining); !word.empty(); word = takeWord(remaining)) {
      if (word == "value") {
        const auto valueFirst = takeWord(remaining);
        if (!valueFirst.empty()) {
          auto valueLast = valueFirst;
          for (auto next = takeWord(remaining); !next.empty(); next = takeWord(remaining)) {
            valueLast = next;
          }
          value = {valueFirst.data(), static_cast<size_t>(valueLast.data() + valueLast.size() - valueFirst.data())};
        }
        break;
      }
      last = word;
    }
    if (onOption_) {
      onOption_({first.data(), static_cast<size_t>(last.data() + last.size() - first.data())}, value);
    }
  }
}
