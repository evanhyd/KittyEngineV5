// Single-threaded successful-probe throughput through Kitty's actual adapter.
// Position parsing and console output are outside the measured regions.
#include "tablebase.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  using namespace bb;
  try {
    if (argc != 6) throw std::invalid_argument("Usage: bench TABLES FENS wdl|root ROUNDS SECONDS");
    const bool root = std::string_view(argv[3]) == "root";
    if (!root && std::string_view(argv[3]) != "wdl") throw std::invalid_argument("Unknown probe mode");
    const int rounds = std::stoi(argv[4]);
    const double seconds = std::stod(argv[5]);
    if (rounds < 1 || seconds < 0.1 || seconds > 60) throw std::invalid_argument("Invalid duration or round count");

    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("Could not open position corpus");
    std::vector<BoardState> positions;
    for (std::string fen; std::getline(input, fen);) {
      positions.emplace_back(fen);
      if (positions.back().getHalfmoveClock() != 0) throw std::invalid_argument("WDL corpus must have zero clocks");
    }
    if (positions.empty()) throw std::invalid_argument("Position corpus is empty");

    tablebase::Service tables;
    const auto initStart = Clock::now();
    tables.setOption("SyzygyPath", argv[1]);
    tables.setOption("SyzygyProbeLimit", "7");
    const double initMs = std::chrono::duration<double, std::milli>(Clock::now() - initStart).count();
    if (!tables.enabled()) throw std::runtime_error("No tablebases discovered");
    std::cerr << "positions=" << positions.size() << " init_ms=" << initMs
      << " msvc=" << _MSC_FULL_VER << '\n';

    struct Sample { uint64_t probes = 0, failures = 0, checksum = 0; };
    const auto pass = [&](Sample& sample) {
      for (const auto& state : positions) {
        if (root) {
          // Include root legal move generation as well as conversion, DTZ
          // probing, matching to engine moves, and selecting the best group.
          MoveList legal;
          if (state.getSideToMove() == White) state.generateMoves<White>(legal);
          else state.generateMoves<Black>(legal);
          const auto result = tables.rankRoot(state, legal, false);
          if (result) sample.checksum += result->bestMoves.size() + result->outcome.value_or(2) + 2;
          else ++sample.failures;
        } else {
          const auto result = tables.probeWdl(state);
          if (result) sample.checksum += static_cast<int>(*result) + 3;
          else ++sample.failures;
        }
        ++sample.probes;
      }
    };

    std::cout << "phase,round,probes,failures,seconds,probes_per_second,checksum\n";
    for (int round = -1; round < rounds; ++round) {
      Sample sample;
      const auto start = Clock::now();
      double elapsed;
      do {
        pass(sample);
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
      } while (round >= 0 && elapsed < seconds);
      std::cout << (round < 0 ? "first_pass" : "warm") << ',' << round << ',' << sample.probes
        << ',' << sample.failures << ',' << elapsed << ',' << (sample.probes - sample.failures) / elapsed
        << ',' << sample.checksum << '\n';
      if (sample.failures) throw std::runtime_error("Corpus contains failed probes; throughput would be misleading");
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
