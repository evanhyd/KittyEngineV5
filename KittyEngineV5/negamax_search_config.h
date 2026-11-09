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
    int32_t aspirationWindow;

    // Reverse Futility Pruning (Static Null Move Pruning)
    int reverseFutilityDepth;
    int32_t reverseFutilityMultiplier;
    int32_t reverseFutilityBase;
    int32_t reverseFutilityImprovingBonus;

    // Null Move Pruning
    int nullMoveDepth;
    int nullMoveBase;
    int nullMoveDepthDivisor;
    int32_t nullMoveEvalDivisor;
    int nullMoveMaxEvalBonus;

    // Futility Pruning
    int futilityDepth;
    int32_t futilityMultiplier;
    int32_t futilityBase;
    int32_t futilityImprovingBonus;

    // SEE Pruning
    int seePruningDepth;
    int32_t seePruningMultiplier;
    int32_t seePruningBase;

    // Late Move Pruning
    int lateMovePruningDepth;
    int lateMovePruningMultiplier;
    int lateMovePruningBase;

    // History Pruning
    int historyPruningDepth;
    int32_t historyPruningMultiplier;
    int32_t historyPruningBase;

    // Late Move Reductions
    int lateMoveReductionDepth;
    double lateMoveReductionBase;
    double lateMoveReductionDivisor;
    double lateMoveReductionHistoryK;
    double lateMoveReductionHistoryB;

    // Singular Extension
    int singularExtensionDepth;
    int singularExtensionDepthMargin;
    int32_t singularExtensionMultiplier;
    int32_t singularExtensionBase;

    // Improving Heuristic
    int32_t improvingThreshold;

    // History Heuristics Limits
    int32_t maxQuietHistory;
    int32_t maxRefutationHistory;
    int32_t maxFollowupHistory;
    int32_t maxCaptureHistory;

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
        "singular_extension:\n"
        "  depth: {}\n"
        "  depth_margin: {}\n"
        "  multiplier: {}\n"
        "  base: {}\n\n"
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
        singularExtensionDepth,
        singularExtensionDepthMargin,
        singularExtensionMultiplier,
        singularExtensionBase,
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
        size_t i = 0, j = 0, k = 0;
        while (i < section.size() || j < key.size()) {
          char c;
          if (i < section.size()) {
            c = section[i++];
          } else {
            c = key[j++];
          }
          if (c == '_' || c == '-' || c == ' ') continue;
          if (k >= targetNormalized.size()) return false;
          if (static_cast<char>(std::tolower(static_cast<unsigned char>(c))) != targetNormalized[k++]) return false;
        }
        return k == targetNormalized.size();
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
        else if (matchesKey(section, rawKey, "reversefutilitypruningdepth")) {
          parseInt(rawVal, reverseFutilityDepth);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningmultiplier")) {
          parseInt(rawVal, reverseFutilityMultiplier);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningbase")) {
          parseInt(rawVal, reverseFutilityBase);
        } else if (matchesKey(section, rawKey, "reversefutilitypruningimprovingbonus")) {
          parseInt(rawVal, reverseFutilityImprovingBonus);
        }
        // NMP
        else if (matchesKey(section, rawKey, "nullmovepruningdepth")) {
          parseInt(rawVal, nullMoveDepth);
        } else if (matchesKey(section, rawKey, "nullmovepruningbase")) {
          parseInt(rawVal, nullMoveBase);
        } else if (matchesKey(section, rawKey, "nullmovepruningdepthdivisor")) {
          parseInt(rawVal, nullMoveDepthDivisor);
        } else if (matchesKey(section, rawKey, "nullmovepruningevaldivisor")) {
          parseInt(rawVal, nullMoveEvalDivisor);
        } else if (matchesKey(section, rawKey, "nullmovepruningmaxevalbonus")) {
          parseInt(rawVal, nullMoveMaxEvalBonus);
        }
        // FP
        else if (matchesKey(section, rawKey, "futilitypruningdepth")) {
          parseInt(rawVal, futilityDepth);
        } else if (matchesKey(section, rawKey, "futilitypruningmultiplier")) {
          parseInt(rawVal, futilityMultiplier);
        } else if (matchesKey(section, rawKey, "futilitypruningbase")) {
          parseInt(rawVal, futilityBase);
        } else if (matchesKey(section, rawKey, "futilitypruningimprovingbonus")) {
          parseInt(rawVal, futilityImprovingBonus);
        }
        // SEE
        else if (matchesKey(section, rawKey, "seepruningdepth")) {
          parseInt(rawVal, seePruningDepth);
        } else if (matchesKey(section, rawKey, "seepruningmultiplier")) {
          parseInt(rawVal, seePruningMultiplier);
        } else if (matchesKey(section, rawKey, "seepruningbase")) {
          parseInt(rawVal, seePruningBase);
        }
        // LMP
        else if (matchesKey(section, rawKey, "latemovepruningdepth")) {
          parseInt(rawVal, lateMovePruningDepth);
        } else if (matchesKey(section, rawKey, "latemovepruningmultiplier")) {
          parseInt(rawVal, lateMovePruningMultiplier);
        } else if (matchesKey(section, rawKey, "latemovepruningbase")) {
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
        else if (matchesKey(section, rawKey, "latemovereductionsdepth")) {
          parseInt(rawVal, lateMoveReductionDepth);
        } else if (matchesKey(section, rawKey, "latemovereductionsbase")) {
          parseDouble(rawVal, lateMoveReductionBase);
        } else if (matchesKey(section, rawKey, "latemovereductionsdivisor")) {
          parseDouble(rawVal, lateMoveReductionDivisor);
        } else if (matchesKey(section, rawKey, "latemovereductionshistoryk")) {
          parseDouble(rawVal, lateMoveReductionHistoryK);
        } else if (matchesKey(section, rawKey, "latemovereductionshistoryb")) {
          parseDouble(rawVal, lateMoveReductionHistoryB);
        }
        // Singular Extension
        else if (matchesKey(section, rawKey, "singularextensiondepth")) {
          parseInt(rawVal, singularExtensionDepth);
        } else if (matchesKey(section, rawKey, "singularextensiondepthmargin")) {
          parseInt(rawVal, singularExtensionDepthMargin);
        } else if (matchesKey(section, rawKey, "singularextensionmultiplier")) {
          parseInt(rawVal, singularExtensionMultiplier);
        } else if (matchesKey(section, rawKey, "singularextensionbase")) {
          parseInt(rawVal, singularExtensionBase);
        }
        // History Limits
        else if (matchesKey(section, rawKey, "historyheuristicsmaxquiethistory")) {
          parseInt(rawVal, maxQuietHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxrefutationhistory")) {
          parseInt(rawVal, maxRefutationHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxfollowuphistory")) {
          parseInt(rawVal, maxFollowupHistory);
        } else if (matchesKey(section, rawKey, "historyheuristicsmaxcapturehistory")) {
          parseInt(rawVal, maxCaptureHistory);
        }
      }
    }
  };
}
