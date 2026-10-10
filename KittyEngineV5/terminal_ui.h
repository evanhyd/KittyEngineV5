#pragma once
#include "boardstate.h"
#include "evaluation_policy.h"
#include "notation.h"
#include "perft_driver.h"
#include "position_fens.h"
#include "time_control_policy.h"
#include "tablebase_uci.h"
#include <build_config.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace bb::user_interface {
  template <typename BoardT>
  class TerminalUI {
    BoardT& board_;
    std::istream& uciInput_;
    std::ostream& uciOutput_;
    std::ostream& humanOutput_;
    std::string printBuffer_;
    std::string lastFrame_;
    bool running_ = false;

    void reportProtocolError(std::string_view message) {
      std::string safeMessage{message};
      std::replace_if(safeMessage.begin(), safeMessage.end(),
                      [](char ch) { return ch == '\r' || ch == '\n'; }, ' ');
      uciOutput_ << std::format("info string error: {}\n", safeMessage);
    }

    void renderState(const BoardState& state) {
      printBuffer_.clear();
      for (Square rank = 0; rank < kBoardLenSize; ++rank) {
        printBuffer_ += "  -------------------------------\n ";
        for (Square file = 0; file < kBoardLenSize; ++file) {
          const auto occupant = state.getPieceAt(rankFileToSquare(rank, file));
          const char symbol = occupant
            ? notation::pieceToAsciiVisual(std::get<0>(*occupant), std::get<1>(*occupant))
            : ' ';
          printBuffer_ += "| ";
          printBuffer_ += symbol;
          printBuffer_ += ' ';
        }
        printBuffer_ += "| ";
        printBuffer_ += static_cast<char>('8' - rank);
        printBuffer_ += '\n';
      }
      printBuffer_ += "  -------------------------------\n   a   b   c   d   e   f   g   h\n";
      printBuffer_ += std::format("Side to move: {}\nCastling: {}\nEn passant: {}\nHalfmove clock: {}\nFullmove number: {}\n",
                                  notation::sideToString(state.getSideToMove()),
                                  notation::castleToString(state.getCastlingRights()),
                                  notation::squareToString(state.getEnpassantSquare()),
                                  state.getHalfmoveClock(), state.getFullmoveNumber());
      printBuffer_ += "FEN: ";
      printBuffer_ += notation::boardToFen(state);
      printBuffer_ += '\n';

      if (printBuffer_ == lastFrame_) {
        return;
      }

      humanOutput_.write(printBuffer_.data(), static_cast<std::streamsize>(printBuffer_.size()));
      humanOutput_.flush();
      if (humanOutput_) {
        lastFrame_ = printBuffer_;
      }
    }

    void render() {
      renderState(board_.getState());
    }

  public:
    TerminalUI(BoardT& board, std::istream& uciInput, std::ostream& uciOutput,
               std::ostream& humanOutput)
      : board_(board), uciInput_(uciInput), uciOutput_(uciOutput),
        humanOutput_(humanOutput) {}

    void run() {
      humanOutput_ << std::format("Welcome to KittyEngineV5.\nBuild Version: {}_{}\n", __DATE__, __TIME__);
      render();

      auto protocol = uci::makeProtocol<tablebase::kEnabled>(
        [this](std::string_view name, std::string_view value) {
          const auto status = board_.setTablebaseOption(name, value);
          if (!status.empty()) {
            uciOutput_ << "info string " << status << '\n';
          }
        },
        [this] {
          uciOutput_ << "id name KittyEngineV5\nid author UnboxTheCat\n";
          if constexpr (tablebase::kEnabled) {
            if (board_.hasTablebaseService()) {
              uciOutput_ << "option name SyzygyPath type string default <empty>\n"
                "option name SyzygyProbeLimit type spin default 5 min 0 max 7\n"
                "option name SyzygyProbeDepth type spin default 1 min 0 max 64\n";
            }
          }
          uciOutput_ << "uciok\n";
        },
        [this] {
          uciOutput_ << "readyok\n";
        },
        [this] {
          board_.reset();
          render();
        },
        [this](std::string_view fen, std::span<const std::string_view> moves) {
          board_.setPosition(fen, moves);
          render();
        },
        [this](std::span<const std::string_view> args) {
          std::optional<int> depth;
          std::optional<int64_t> whiteTime;
          std::optional<int64_t> blackTime;
          std::optional<int64_t> whiteIncrement;
          std::optional<int64_t> blackIncrement;
          std::optional<int64_t> movesToGo;
          for (size_t i = 0; i < args.size(); ++i) {
            if (++i == args.size()) {
              throw std::invalid_argument("go option needs a value");
            }
            const std::string_view value = args[i];
            int64_t parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (error != std::errc{} || end != value.data() + value.size() || parsed < 0) {
              throw std::invalid_argument("go option needs a nonnegative integer");
            }
            const std::string_view option = args[i - 1];
            if (option == "depth") {
              if (depth || parsed == 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("go depth needs a positive integer");
              }
              depth = static_cast<int>(parsed);
            } else {
              std::optional<int64_t>* target = nullptr;
              if (option == "wtime") target = &whiteTime;
              else if (option == "btime") target = &blackTime;
              else if (option == "winc") target = &whiteIncrement;
              else if (option == "binc") target = &blackIncrement;
              else if (option == "movestogo") target = &movesToGo;
              else throw std::invalid_argument("unsupported go option");
              if (*target) {
                throw std::invalid_argument("duplicate go option");
              }
              *target = parsed;
            }
          }
          const bool hasClock = whiteTime || blackTime;
          if (hasClock != (whiteTime && blackTime) ||
              (!hasClock && (whiteIncrement || blackIncrement))) {
            throw std::invalid_argument("go time control needs wtime and btime");
          }
          if (!depth && !hasClock) {
            throw std::invalid_argument("go depth or time control is required");
          }

          std::optional<time_control::TimeControl> timeControl;
          if (hasClock) {
            if (movesToGo && *movesToGo > std::numeric_limits<int>::max()) {
              throw std::invalid_argument("go movestogo is too large");
            }
            timeControl.emplace();
            timeControl->wtime = std::chrono::milliseconds{*whiteTime};
            timeControl->btime = std::chrono::milliseconds{*blackTime};
            if (whiteIncrement) timeControl->winc = std::chrono::milliseconds{*whiteIncrement};
            if (blackIncrement) timeControl->binc = std::chrono::milliseconds{*blackIncrement};
            if (movesToGo) timeControl->movesToGo = static_cast<int>(*movesToGo);
          }

          int completedDepth = 0;
          uint64_t totalNodes = 0;
          uint64_t totalHits = 0;
          const auto searchStart = std::chrono::steady_clock::now();
          const int searchDepth = std::min(depth.value_or(searching::kMaxDepthHardCutoff), searching::kMaxDepthHardCutoff);
          const auto result = board_.search(searchDepth, timeControl, [this, &completedDepth, &totalNodes, searchStart, &totalHits](const auto& result) {
            ++completedDepth;
            totalNodes += result.nodesSearched;
            const auto elapsed = std::chrono::steady_clock::now() - searchStart;
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            const double elapsedSeconds = std::max(1e-9, std::chrono::duration<double>(elapsed).count());
            const uint64_t nps = static_cast<uint64_t>(totalNodes / elapsedSeconds);

            constexpr int64_t mateScore = -static_cast<int64_t>(evaluation::kCheckmateScore);
            constexpr int64_t mateWindow = 100;
            const int64_t score = static_cast<int64_t>(result.score);
            const int64_t magnitude = std::abs(score);
            const bool isMateScore = magnitude >= mateScore - mateWindow && magnitude <= mateScore;
            std::string scoreText;
            if (isMateScore) {
              const int64_t movesToMate = (mateScore - magnitude + 1) / 2;
              scoreText = std::format("mate {}", score < 0 ? -movesToMate : movesToMate);
            } else {
              scoreText = std::format("cp {}", score);
            }
            std::string infoLine = std::format("info depth {} score {} time {} nodes {} nps {}",
                                               completedDepth, scoreText, elapsedMs, totalNodes, nps);
            if constexpr (tablebase::kEnabled) {
              totalHits += board_.tablebaseHits();
              std::format_to(std::back_inserter(infoLine), " tbhits {}", totalHits);
            }
            const auto& pv = *result.pvLine;
            if (!pv.empty()) {
              infoLine += " pv";
              for (const Move& move : pv) {
                std::format_to(std::back_inserter(infoLine), " {}", notation::moveToString(move));
              }
            }
            infoLine += '\n';
            uciOutput_.write(infoLine.data(), static_cast<std::streamsize>(infoLine.size()));
            uciOutput_.flush();
          });
          uciOutput_ << std::format("bestmove {}\n",
                                    result.bestMove ? notation::moveToString(*result.bestMove) : "0000");
        },
        [this] {
          running_ = false;
        },
        [this](std::string_view move) {
          board_.playMove(move);
          render();
        },
        [this](uint32_t depth, bool detail) {
          if (detail) {
            constexpr perft::Config config{false, false, true};
            perft::runPerft<config>(board_.getState(), depth, uciOutput_);
          } else {
            constexpr perft::Config config{false, true, false};
            perft::runPerft<config>(board_.getState(), depth, uciOutput_);
          }
        }
      );

      running_ = true;
      for (std::string line; running_ && std::getline(uciInput_, line); ) {
        try {
          protocol.send(line);
        } catch (const std::exception& error) {
          reportProtocolError(error.what());
        }
        uciOutput_.flush();
      }
    }
  };

  template <typename BoardT>
  TerminalUI(BoardT&, std::istream&, std::ostream&, std::ostream&) -> TerminalUI<BoardT>;
}
