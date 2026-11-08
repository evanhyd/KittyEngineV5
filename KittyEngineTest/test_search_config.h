#pragma once
#include "negamax_search_config.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

namespace bb::testing_support {
  inline searching::NegamaxSearchConfig makeTestSearchConfig() {
    searching::NegamaxSearchConfig config{};

    const auto yamlPath = std::filesystem::path(__FILE__).parent_path().parent_path()
      / "KittyEngineV5" / "search_config.yaml";
    std::ifstream file{yamlPath};
    if (file.is_open()) {
      config.deserialize(file);
      return config;
    }

    constexpr std::string_view baselineYaml = R"(
aspiration_window: 80
improving_threshold: 20

reverse_futility_pruning:
  depth: 3
  multiplier: 100
  base: 0
  improving_bonus: 30

null_move_pruning:
  depth: 4
  base: 3
  depth_divisor: 6
  eval_divisor: 200
  max_eval_bonus: 2

futility_pruning:
  depth: 3
  multiplier: 100
  base: 0
  improving_bonus: 20

see_pruning:
  depth: 3
  multiplier: -100
  base: 0

late_move_pruning:
  depth: 4
  multiplier: 2
  base: 6

history_pruning:
  depth: 3
  multiplier: -300
  base: -500

late_move_reductions:
  depth: 3
  base: 0.7844
  divisor: 2.69
  history_k: 0.001
  history_b: 0.0

history_heuristics:
  max_quiet_history: 770
  max_refutation_history: 1230
  max_followup_history: 500
  max_capture_history: 400
)";
    std::istringstream fallbackStream{std::string(baselineYaml)};
    config.deserialize(fallbackStream);
    return config;
  }
}
