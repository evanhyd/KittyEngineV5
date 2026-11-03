#pragma once
#include <cstdint>
#include <charconv>
#include <format>
#include <istream>
#include <ostream>
#include <string>
#include <string_view>

namespace bb::searching {
  struct NegamaxSearchConfig {
    // Aspiration Window
    int32_t aspirationWindow = 60;

    // Reverse Futility Pruning (Static Null Move Pruning)
    int reverseFutilityDepth = 3;
    int32_t reverseFutilityMultiplier = 100;
    int32_t reverseFutilityBase = 0;
    int32_t reverseFutilityImprovingBonus = 30;

    // Null Move Pruning
    int nullMoveDepth = 4;
    int nullMoveBase = 3;
    int nullMoveDepthDivisor = 6;
    int32_t nullMoveEvalDivisor = 200;
    int nullMoveMaxEvalBonus = 2;

    // Futility Pruning
    int futilityDepth = 3;
    int32_t futilityMultiplier = 100;
    int32_t futilityBase = 0;
    int32_t futilityImprovingBonus = 20;

    // SEE Pruning
    int seePruningDepth = 3;
    int32_t seePruningMultiplier = -100;
    int32_t seePruningBase = 0;

    // Late Move Pruning
    int lateMovePruningDepth = 4;
    int lateMovePruningMultiplier = 2;
    int lateMovePruningBase = 6;

    // History Pruning
    int historyPruningDepth = 3;
    int32_t historyPruningMultiplier = -300;
    int32_t historyPruningBase = -500;

    // Late Move Reductions
    int lateMoveReductionDepth = 3;
    double lateMoveReductionBase = 0.7844;
    double lateMoveReductionDivisor = 2.69;
    double lateMoveReductionHistoryK = 1.0 / 1000.0;
    double lateMoveReductionHistoryB = 0.0;

    // Improving Heuristic
    int32_t improvingThreshold = 20;

    // History Heuristics Limits
    int32_t maxQuietHistory = 770;
    int32_t maxRefutationHistory = 1230;
    int32_t maxFollowupHistory = 500;
    int32_t maxCaptureHistory = 400;

    // Safety and Stability Invariants
    static constexpr int kNullMoveMinPieces = 2;

    void serialize(std::ostream& stream) const {
      std::string out;
      out.reserve(2048);
      std::format_to(std::back_inserter(out),
        "# KittyEngine Negamax Search Configuration\n"
        "aspiration_window: {}\n"
        "improving_threshold: {}\n\n"
        "reverse_futility_pruning:\n"
        "  depth: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n"
        "  improving_bonus: {}\n\n"
        "null_move_pruning:\n"
        "  depth: {}\n"
        "  base: {}\n"
        "  depth_divisor: {}\n"
        "  eval_divisor: {}\n"
        "  max_eval_bonus: {}\n\n"
        "futility_pruning:\n"
        "  depth: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n"
        "  improving_bonus: {}\n\n"
        "see_pruning:\n"
        "  depth: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n\n"
        "late_move_pruning:\n"
        "  depth: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n\n"
        "history_pruning:\n"
        "  depth: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n\n"
        "late_move_reductions:\n"
        "  depth: {}\n"
        "  base: {}\n"
        "  divisor: {}\n"
        "  history_k: {}\n"
        "  history_b: {}\n\n"
        "history_heuristics:\n"
        "  max_quiet_history: {}\n"
        "  max_refutation_history: {}\n"
        "  max_followup_history: {}\n"
        "  max_capture_history: {}\n",
        aspirationWindow,
        improvingThreshold,
        reverseFutilityDepth,
        reverseFutilityMultiplier,
        reverseFutilityBase,
        reverseFutilityImprovingBonus,
        nullMoveDepth,
        nullMoveBase,
        nullMoveDepthDivisor,
        nullMoveEvalDivisor,
        nullMoveMaxEvalBonus,
        futilityDepth,
        futilityMultiplier,
        futilityBase,
        futilityImprovingBonus,
        seePruningDepth,
        seePruningMultiplier,
        seePruningBase,
        lateMovePruningDepth,
        lateMovePruningMultiplier,
        lateMovePruningBase,
        historyPruningDepth,
        historyPruningMultiplier,
        historyPruningBase,
        lateMoveReductionDepth,
        lateMoveReductionBase,
        lateMoveReductionDivisor,
        lateMoveReductionHistoryK,
        lateMoveReductionHistoryB,
        maxQuietHistory,
        maxRefutationHistory,
        maxFollowupHistory,
        maxCaptureHistory
      );
      stream << out;
    }

    void deserialize(std::istream& stream) {
      auto trim = [](std::string_view s) noexcept -> std::string_view {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
          s.remove_prefix(1);
        }
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
          s.remove_suffix(1);
        }
        return s;
      };

      auto parseInt = [](std::string_view s, auto& out) noexcept {
        s = s.substr(0, s.find_first_of(" \t\r\n#"));
        if (!s.empty()) {
          std::from_chars(s.data(), s.data() + s.size(), out);
        }
      };

      auto parseDouble = [](std::string_view s, double& out) noexcept {
        s = s.substr(0, s.find_first_of(" \t\r\n#"));
        if (!s.empty()) {
          std::from_chars(s.data(), s.data() + s.size(), out);
        }
      };

      auto matchesKey = [](std::string_view section, std::string_view key, std::string_view targetNormalized) noexcept -> bool {
        auto matches = [](std::string_view prefix, std::string_view name, std::string_view target) noexcept -> bool {
          size_t i = 0, j = 0, k = 0;
          while (i < prefix.size() || j < name.size()) {
            char c;
            if (i < prefix.size()) {
              c = prefix[i++];
            } else {
              c = name[j++];
            }
            if (c == '_' || c == '-' || c == ' ') continue;
            if (k >= target.size()) return false;
            if (static_cast<char>(std::tolower(static_cast<unsigned char>(c))) != target[k++]) return false;
          }
          return k == target.size();
        };
        return matches(section, key, targetNormalized) || matches({}, key, targetNormalized);
      };

      std::string line;
      std::string currentSection;
      while (std::getline(stream, line)) {
        std::string_view sv = line;
        const size_t commentPos = sv.find('#');
        if (commentPos != std::string_view::npos) {
          sv = sv.substr(0, commentPos);
        }
        sv = trim(sv);
        if (sv.empty()) continue;

        const size_t colonPos = sv.find(':');
        if (colonPos == std::string_view::npos) continue;

        const std::string_view rawKey = trim(sv.substr(0, colonPos));
        const std::string_view rawVal = trim(sv.substr(colonPos + 1));

        if (rawVal.empty()) {
          currentSection = rawKey;
          continue;
        }

        const std::string_view section = currentSection;

        // Top-level
        if (matchesKey(section, rawKey, "aspirationwindow")) {
          parseInt(rawVal, aspirationWindow);
        } else if (matchesKey(section, rawKey, "improvingthreshold")) {
          parseInt(rawVal, improvingThreshold);
        }
        // RFP
        else if (matchesKey(section, rawKey, "reversefutilitypruningdepth") || matchesKey(section, rawKey, "reversefutilitydepth")) {
          parseInt(rawVal, reverseFutilityDepth);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningmultiplier") || matchesKey(section, rawKey, "reversefutilitymultiplier")) {
          parseInt(rawVal, reverseFutilityMultiplier);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningbase") || matchesKey(section, rawKey, "reversefutilitybase")) {
          parseInt(rawVal, reverseFutilityBase);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningimprovingbonus") || matchesKey(section, rawKey, "reversefutilityimprovingbonus")) {
          parseInt(rawVal, reverseFutilityImprovingBonus);
        }
        // NMP
        else if (matchesKey(section, rawKey, "nullmovepruningdepth") || matchesKey(section, rawKey, "nullmovedepth")) {
          parseInt(rawVal, nullMoveDepth);
        } else if (matchesKey(section, rawKey, "nullmovepruningbase") || matchesKey(section, rawKey, "nullmovebase")) {
          parseInt(rawVal, nullMoveBase);
        } else if (matchesKey(section, rawKey, "nullmovepruningdepthdivisor") || matchesKey(section, rawKey, "nullmovedepthdivisor")) {
          parseInt(rawVal, nullMoveDepthDivisor);
        } else if (matchesKey(section, rawKey, "nullmovepruningevaldivisor") || matchesKey(section, rawKey, "nullmoveevaldivisor")) {
          parseInt(rawVal, nullMoveEvalDivisor);
        } else if (matchesKey(section, rawKey, "nullmovepruningmaxevalbonus") || matchesKey(section, rawKey, "nullmovemaxevalbonus")) {
          parseInt(rawVal, nullMoveMaxEvalBonus);
        }
        // FP
        else if (matchesKey(section, rawKey, "futilitypruningdepth") || matchesKey(section, rawKey, "futilitydepth")) {
          parseInt(rawVal, futilityDepth);
        } else if (matchesKey(section, rawKey, "futilitypruningmultiplier") || matchesKey(section, rawKey, "futilitymultiplier")) {
          parseInt(rawVal, futilityMultiplier);
        } else if (matchesKey(section, rawKey, "futilitypruningbase") || matchesKey(section, rawKey, "futilitybase")) {
          parseInt(rawVal, futilityBase);
        } else if (matchesKey(section, rawKey, "futilitypruningimprovingbonus") || matchesKey(section, rawKey, "futilityimprovingbonus")) {
          parseInt(rawVal, futilityImprovingBonus);
        }
        // SEE
        else if (matchesKey(section, rawKey, "seepruningdepth") || matchesKey(section, rawKey, "seedepth")) {
          parseInt(rawVal, seePruningDepth);
        } else if (matchesKey(section, rawKey, "seepruningmultiplier") || matchesKey(section, rawKey, "seemultiplier")) {
          parseInt(rawVal, seePruningMultiplier);
        } else if (matchesKey(section, rawKey, "seepruningbase") || matchesKey(section, rawKey, "seebase")) {
          parseInt(rawVal, seePruningBase);
        }
        // LMP
        else if (matchesKey(section, rawKey, "latemovepruningdepth") || matchesKey(section, rawKey, "latemovedepth")) {
          parseInt(rawVal, lateMovePruningDepth);
        } else if (matchesKey(section, rawKey, "latemovepruningmultiplier") || matchesKey(section, rawKey, "latemovemultiplier")) {
          parseInt(rawVal, lateMovePruningMultiplier);
        } else if (matchesKey(section, rawKey, "latemovepruningbase") || matchesKey(section, rawKey, "latemovebase")) {
          parseInt(rawVal, lateMovePruningBase);
        }
        // History Pruning
        else if (matchesKey(section, rawKey, "historypruningdepth")) {
          parseInt(rawVal, historyPruningDepth);
        } else if (matchesKey(section, rawKey, "historypruningmultiplier")) {
          parseInt(rawVal, historyPruningMultiplier);
        } else if (matchesKey(section, rawKey, "historypruningbase")) {
          parseInt(rawVal, historyPruningBase);
        }
        // LMR
        else if (matchesKey(section, rawKey, "latemovereductionsdepth") || matchesKey(section, rawKey, "lmrdepth")) {
          parseInt(rawVal, lateMoveReductionDepth);
        } else if (matchesKey(section, rawKey, "latemovereductionsbase") || matchesKey(section, rawKey, "lmrbase")) {
          parseDouble(rawVal, lateMoveReductionBase);
        } else if (matchesKey(section, rawKey, "latemovereductionsdivisor") || matchesKey(section, rawKey, "lmrdivisor")) {
          parseDouble(rawVal, lateMoveReductionDivisor);
        } else if (matchesKey(section, rawKey, "latemovereductionshistoryk") || matchesKey(section, rawKey, "lmrhistoryk")) {
          parseDouble(rawVal, lateMoveReductionHistoryK);
        } else if (matchesKey(section, rawKey, "latemovereductionshistoryb") || matchesKey(section, rawKey, "lmrhistoryb")) {
          parseDouble(rawVal, lateMoveReductionHistoryB);
        }
        // History Limits
        else if (matchesKey(section, rawKey, "historyheuristicsmaxquiethistory") || matchesKey(section, rawKey, "maxquiethistory")) {
          parseInt(rawVal, maxQuietHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxrefutationhistory") || matchesKey(section, rawKey, "maxrefutationhistory")) {
          parseInt(rawVal, maxRefutationHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxfollowuphistory") || matchesKey(section, rawKey, "maxfollowuphistory")) {
          parseInt(rawVal, maxFollowupHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxcapturehistory") || matchesKey(section, rawKey, "maxcapturehistory")) {
          parseInt(rawVal, maxCaptureHistory);
        }
      }
    }
  };
}
